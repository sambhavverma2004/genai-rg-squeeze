"""Fine-tune a causal LM with LoRA under a rank-annealing policy, then evaluate.

Example (one run, ~1-2 GPU hours scale depends on model / steps; measure before launching a grid):
  python llm_experiment.py --model Qwen/Qwen2.5-1.5B --task boolq --policy rg --r-src 64 --r-tgt 4 --steps 2000

Tasks: boolq, arc_easy, arc_challenge, openbookqa  (multiple-choice, scored by next-token logprob of the
answer symbol)   and   gsm8k  (generative, greedy decoding, exact match on the final number).
Dataset field names are the standard HF ones; if a loader errors on first use, print one example and adjust
the small `load_*` function - everything else is task-agnostic.
"""
import argparse, json, os, random, re, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from rgsqueeze import inject_lora, make_policy, train, current_rank
from rgsqueeze.lora import lora_parameters

LETTERS = ["A", "B", "C", "D", "E"]


# ----------------------------------------------------------------------------- data
def load_boolq(split, n):
    from datasets import load_dataset
    d = load_dataset("google/boolq", split="train" if split == "train" else "validation")
    ex = [{"prompt": f"Passage: {r['passage']}\nQuestion: {r['question']}?\nAnswer (yes or no):",
           "cands": [" yes", " no"], "label": 0 if r["answer"] else 1} for r in d]
    return ex[:n] if n else ex


def _load_mc(name, config, split, n, stem_key):
    from datasets import load_dataset
    d = load_dataset(name, config, split=split)
    ex = []
    for r in d:
        labels, texts, key = list(r["choices"]["label"]), r["choices"]["text"], r["answerKey"]
        if all(l.isdigit() for l in labels):  # some ARC items use 1-5; " 1"/" 2" share a first token in Qwen's tokenizer
            labels = [LETTERS[int(l) - 1] for l in labels]
            key = LETTERS[int(key) - 1] if key.isdigit() else key
        if key not in labels:
            continue
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        ex.append({"prompt": f"Question: {r[stem_key]}\n{opts}\nAnswer:",
                   "cands": [f" {l}" for l in labels], "label": labels.index(key)})
    return ex[:n] if n else ex


def load_task(task, split, n=None):
    s = "train" if split == "train" else ("validation" if task != "openbookqa" else "validation")
    if task == "boolq":
        return load_boolq(split, n)
    if task == "arc_easy":
        return _load_mc("allenai/ai2_arc", "ARC-Easy", s, n, "question")
    if task == "arc_challenge":
        return _load_mc("allenai/ai2_arc", "ARC-Challenge", s, n, "question")
    if task == "openbookqa":
        return _load_mc("allenai/openbookqa", "main", s, n, "question_stem")
    if task == "gsm8k":
        from datasets import load_dataset
        d = load_dataset("openai/gsm8k", "main", split="train" if split == "train" else "test")
        ex = [{"prompt": f"Question: {r['question']}\nAnswer:", "target": " " + r["answer"],
               "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for r in d]
        return ex[:n] if n else ex
    raise ValueError(task)


def build_train_item(tok, ex, max_len):
    target = ex["target"] if "target" in ex else ex["cands"][ex["label"]]
    p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if "target" in ex else [])
    p = p[-(max_len - len(t)):]
    return p + t, [-100] * len(p) + t


def make_batcher(tok, examples, bs, max_len, device, seed):
    rng = random.Random(seed)
    items = [build_train_item(tok, e, max_len) for e in examples]
    order = []
    def get():
        nonlocal order
        if len(order) < bs:
            order = list(range(len(items))); rng.shuffle(order)
        idx = [order.pop() for _ in range(bs)]
        L = max(len(items[i][0]) for i in idx)
        pad = tok.pad_token_id
        ids = torch.tensor([items[i][0] + [pad] * (L - len(items[i][0])) for i in idx])
        lab = torch.tensor([items[i][1] + [-100] * (L - len(items[i][1])) for i in idx])
        att = torch.tensor([[1] * len(items[i][0]) + [0] * (L - len(items[i][0])) for i in idx])
        return ids.to(device), att.to(device), lab.to(device)
    return get


def lm_loss(model, batch):
    ids, att, lab = batch
    return model(input_ids=ids, attention_mask=att, labels=lab).loss


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def eval_mc(model, tok, examples, device, bs=16, max_len=1024):
    model.eval(); correct = 0
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        enc = [tok(e["prompt"], add_special_tokens=False)["input_ids"][-max_len:] for e in chunk]
        L = max(map(len, enc))
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in enc]).to(device)
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in enc]).to(device)
        logits = model(input_ids=ids, attention_mask=att).logits
        for j, e in enumerate(chunk):
            last = logits[j, len(enc[j]) - 1].float()
            cand_ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in e["cands"]]
            assert len(set(cand_ids)) == len(cand_ids), f"candidates share a first token: {e['cands']}"
            correct += int(max(range(len(cand_ids)), key=lambda k: last[cand_ids[k]].item()) == e["label"])
    model.train()
    return correct / len(examples)


@torch.no_grad()
def eval_gsm8k(model, tok, examples, device, bs=16, max_new=256):
    model.eval(); correct = 0
    tok.padding_side = "left"
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        enc = tok([e["prompt"] for e in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for e, o in zip(chunk, out):
            txt = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"####\s*(-?[\d,\.]+)", txt)
            pred = m.group(1).replace(",", "").rstrip(".") if m else None
            correct += int(pred is not None and pred == e["gold"])
    tok.padding_side = "right"; model.train()
    return correct / len(examples)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--task", default="boolq")
    ap.add_argument("--policy", default="rg", choices=["direct", "post", "cont", "in_std", "in_min", "rg"])
    ap.add_argument("--r-src", type=int, default=64)
    ap.add_argument("--r-tgt", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--scale", type=float, default=2.0, help="LoRA output scale (constant across squeezes)")
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-steps", type=int, default=200)
    ap.add_argument("--cont-steps", type=int, default=200)
    ap.add_argument("--tau0", type=float, default=0.98)
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--max-shrink", type=int, default=4)
    ap.add_argument("--out", default="results/llm")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    n = inject_lora(model, a.r_src if a.policy != "direct" else a.r_tgt, a.targets.split(","), a.scale)
    print(f"injected LoRA into {n} linear layers; trainable={sum(p.numel() for p in lora_parameters(model))/1e6:.2f}M", flush=True)

    train_ex = load_task(a.task, "train", n=20000)
    test_ex = load_task(a.task, "test", n=a.eval_n)
    get_batch = make_batcher(tok, train_ex, a.bs, a.max_len, dev, a.seed)
    pol = make_policy(a.policy, a.steps, a.r_src, a.r_tgt, min_steps=a.min_steps, cont_steps=a.cont_steps,
                      **({"tau0": a.tau0, "quantile": a.quantile, "max_shrink": a.max_shrink} if a.policy == "rg" else {}))
    ac = {"device_type": "cuda", "dtype": torch.bfloat16} if dev == "cuda" else None
    ev = (lambda m: eval_gsm8k(m, tok, test_ex[:250], dev)) if a.task == "gsm8k" else (lambda m: eval_mc(m, tok, test_ex, dev))

    t0 = time.time()
    print(f"device={dev}; training {a.steps} steps...", flush=True)
    hist = train(model, pol, get_batch, lm_loss, a.steps, a.lr, autocast=ac, log_every=10)
    if a.policy == "post":
        from rgsqueeze import squeeze_all
        squeeze_all(model, a.r_tgt)
    train_time = time.time() - t0
    t1 = time.time()
    acc = ev(model)
    eval_time = time.time() - t1
    res = dict(vars(a), final_rank=current_rank(model), acc=acc, train_time_s=train_time, eval_time_s=eval_time,
               rank_traj=hist["rank"][::25], loss_tail=sum(hist["loss"][-50:]) / 50,
               squeeze_log=getattr(pol, "log", []))
    os.makedirs(a.out, exist_ok=True)
    fn = f"{a.out}/{a.task}_{a.policy}_r{a.r_tgt}_s{a.seed}.json"
    json.dump(res, open(fn, "w"), indent=1)
    print(f"[{a.task}/{a.policy} r_src={a.r_src}->r_tgt={a.r_tgt}] acc={acc:.4f} final_rank={res['final_rank']} -> {fn}")


if __name__ == "__main__":
    main()
