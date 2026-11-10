"""CPU sanity study (NOT a benchmark): multi-module linear teacher-student with controlled spectra.

Each of L modules must learn dW*_l = U diag(i^-p_l) V^T from noisy samples y = dW*_l x + eps, with
anisotropic inputs x ~ N(0, Sigma). p_l spans 0.3 (flat spectrum, hard to compress) .. 2.5 (steep).
This isolates the *schedule* question: given a fixed step budget and a deployment rank r_tgt, which
rank-annealing policy ends closest to the teacher? Metric = Sigma-weighted excess loss
tr(D Sigma D^T)/tr(dW* Sigma dW*^T), D = dW_hat - dW*, averaged over modules (lower is better).
Learning rate is tuned per (policy, r_tgt) on separate tuning seeds, then reported on fresh seeds.
"""
import json, sys, time, torch, torch.nn as nn
sys.path.insert(0, ".")
from rgsqueeze import ReRankLoRALinear, make_policy, train
from rgsqueeze.lora import lora_modules

D, L, BS, T, R_SRC, NOISE = 128, 8, 64, 2000, 64, 0.05
P = torch.linspace(0.3, 2.5, L)

def make_problem(seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    lam = (torch.arange(1, D + 1).float()) ** -0.7           # input covariance spectrum (cond ~ 50)
    Sigma = (Q * lam) @ Q.t()
    chol = Q * lam.sqrt()
    W = []
    for p in P:
        U, _ = torch.linalg.qr(torch.randn(D, D, generator=g)); V, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
        s = torch.arange(1, D + 1).float() ** -float(p)
        W.append((U * s) @ V.t() / s.pow(2).sum().sqrt() * 3.0)
    return Sigma, chol, W

class Toy(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.layers = nn.ModuleList([ReRankLoRALinear(nn.Linear(D, D, bias=False), rank) for _ in range(L)])
        for l in self.layers: nn.init.zeros_(l.base.weight)

def run(policy_name, r_tgt, lr, seed, record=False):
    torch.manual_seed(seed)
    Sigma, chol, W = make_problem(seed)
    pol = make_policy(policy_name, T, R_SRC, r_tgt, min_steps=200, cont_steps=300)
    model = Toy(pol.start_rank())
    g = torch.Generator().manual_seed(seed + 999)
    def get_batch():
        x = torch.randn(BS, D, generator=g) @ chol.t()
        return x, [x @ w.t() + NOISE * torch.randn(BS, D, generator=g) for w in W]
    def loss_fn(m, batch):
        x, ys = batch
        return sum(((l(x) - y) ** 2).mean() for l, y in zip(m.layers, ys)) / L
    h = train(model, pol, get_batch, loss_fn, T, lr, warmup=50, rewarm=50)
    if policy_name == "post":                                  # Post-Squeeze: compress after the last step
        for m in lora_modules(model): m.squeeze(r_tgt)
    with torch.no_grad():
        errs = []
        for l, w in zip(lora_modules(model), W):
            d = l.delta_w() - w
            errs.append(float((d @ Sigma * d).sum() / (w @ Sigma * w).sum()))
    out = {"err": sum(errs) / L, "per_layer": errs}
    if record: out["log"] = getattr(pol, "log", []); out["rank_traj"] = h["rank"][::50]
    return out

if __name__ == "__main__":
    LRS = [3e-3, 1e-2, 3e-2]
    POLS = ["direct", "post", "cont", "in_std", "in_min", "rg"]
    TUNE, EVAL = [100, 101], [0, 1, 2, 3, 4]
    res, t0 = {}, time.time()
    r_list = [int(a) for a in sys.argv[1:]] or [1, 4, 8]
    torch.set_num_threads(1)
    for r_tgt in r_list:
        for pol in POLS:
            tune = {lr: sum(run(pol, r_tgt, lr, s)["err"] for s in TUNE) / len(TUNE) for lr in LRS}
            best = min(tune, key=tune.get)
            runs = [run(pol, r_tgt, best, s, record=(s == 0)) for s in EVAL]
            errs = torch.tensor([r["err"] for r in runs])
            res[f"{pol}|{r_tgt}"] = {"lr": best, "mean": errs.mean().item(), "sd": errs.std().item(),
                                      "per_layer": torch.tensor([r["per_layer"] for r in runs]).mean(0).tolist(),
                                      "log": runs[0].get("log"), "rank_traj": runs[0].get("rank_traj")}
            print(f"r_tgt={r_tgt:>2} {pol:7s} lr={best:<6g} err={errs.mean():.4f} ± {errs.std():.4f}   [{time.time()-t0:.0f}s]", flush=True)
    json.dump({"cfg": dict(D=D, L=L, BS=BS, T=T, R_SRC=R_SRC, NOISE=NOISE, P=P.tolist(), tune=TUNE, eval=EVAL), "res": res},
              open("results/toy_results_%s.json" % "_".join(map(str, r_list)), "w"), indent=1)
