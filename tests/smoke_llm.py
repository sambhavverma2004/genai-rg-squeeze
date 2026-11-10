"""End-to-end smoke test of llm_experiment plumbing on a random tiny Llama + char tokenizer (no downloads)."""
import sys, random, torch
sys.path.insert(0, ".")
from transformers import LlamaConfig, LlamaForCausalLM
import llm_experiment as E
from rgsqueeze import inject_lora, make_policy, train, current_rank, squeeze_all

class CharTok:
    pad_token_id, eos_token_id, eos_token, pad_token, padding_side = 0, 1, "<e>", "<p>", "right"
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 120) for c in text]}

def make_examples(n, seed):
    r = random.Random(seed); ex = []
    for _ in range(n):
        c = r.choice("abcdefghijklmnopqrstuvwxyz")
        ex.append({"prompt": f"Q: {c}{c}\nAnswer:", "cands": ["y", "n"], "label": 0 if c <= "m" else 1})
    return ex

def run(policy, r_tgt, steps=500):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=130, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, initializer_range=0.2)
    model = LlamaForCausalLM(cfg)
    for p in model.parameters(): p.requires_grad_(False)
    r_src = 16
    n = inject_lora(model, r_src if policy != "direct" else r_tgt, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 2.0)
    assert n == 14, n
    tok = CharTok()
    tr, te = make_examples(2000, 1), make_examples(300, 2)
    gb = E.make_batcher(tok, tr, 16, 32, "cpu", 0)
    pol = make_policy(policy, steps, r_src, r_tgt, min_steps=50, cont_steps=50)
    hist = train(model, pol, gb, E.lm_loss, steps, 1e-2)
    if policy == "post": squeeze_all(model, r_tgt)
    return current_rank(model), E.eval_mc(model, tok, te, "cpu"), hist["loss"][-1], getattr(pol, "log", [])

if __name__ == "__main__":
    for pol in ("direct", "post", "in_min", "rg"):
        rank, acc, loss, log = run(pol, 2)
        assert rank == 2, (pol, rank)
        print(f"{pol:7s} final_rank={rank} acc={acc:.3f} last_loss={loss:.3f} squeezes={len(log)}")
