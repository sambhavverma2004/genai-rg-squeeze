import time
import torch
from .lora import lora_parameters, squeeze_all, all_spectra, current_rank


def train(model, policy, get_batch, loss_fn, total_steps, lr, warmup=50, rewarm=50,
          weight_decay=0.0, eval_fn=None, eval_every=0, grad_clip=1.0, autocast=None, log_every=0):
    """Generic loop. After every squeeze the optimizer is rebuilt (fresh Adam moments, because the
    factor bases changed) and the LR is re-warmed over `rewarm` steps. Every squeeze policy gets
    this identical treatment, so comparisons are fair. Returns a history dict."""
    r0 = policy.start_rank()
    # Build the model at policy.start_rank(): squeezing a fresh LoRA (B=0) would zero both factors.
    assert current_rank(model) == r0, f"model rank {current_rank(model)} != policy start rank {r0}"
    opt = torch.optim.AdamW(lora_parameters(model), lr=lr, weight_decay=weight_decay)
    hist = {"loss": [], "rank": [], "evals": []}
    since_reset, ramp = 0, warmup
    t0 = time.time()
    for step in range(total_steps + 1):
        new_r = policy.decide(step, lambda: all_spectra(model))
        if new_r is not None and new_r < current_rank(model):
            squeeze_all(model, new_r)
            opt = torch.optim.AdamW(lora_parameters(model), lr=lr, weight_decay=weight_decay)
            since_reset, ramp = 0, rewarm
        if step == total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (since_reset + 1) / max(1, ramp))
        batch = get_batch()
        if autocast is not None:
            with torch.autocast(**autocast):
                loss = loss_fn(model, batch)
        else:
            loss = loss_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), grad_clip)
        opt.step()
        since_reset += 1
        hist["loss"].append(float(loss.detach()))
        hist["rank"].append(current_rank(model))
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step+1}/{total_steps}  rank={current_rank(model)}  loss={sum(hist['loss'][-log_every:])/log_every:.4f}  {(step+1)/(time.time()-t0):.2f} it/s", flush=True)
        if eval_fn and eval_every and (step + 1) % eval_every == 0:
            hist["evals"].append((step + 1, current_rank(model), eval_fn(model)))
    return hist
