"""Rank-schedule policies. Every policy sees the same total step budget T.

decide(step, spectra_fn) -> new_rank or None. `spectra_fn()` lazily returns the list of
per-module singular-value tensors (only the adaptive policy calls it).

Baselines re-implement the schedules described in LoRA-Squeeze (arXiv 2602.10993):
  direct   : plain LoRA at r_tgt
  post     : train at r_src, squeeze once at the very end (Post-Squeeze)
  cont     : squeeze to r_tgt at T-M, continue M steps (Cont-Squeeze, budget-matched)
  in_std   : halving schedule, stage steps proportional to stage rank (In-Squeeze, Standard)
  in_min   : as in_std but every stage first gets a floor of `min_steps` (In-Squeeze, Min Steps)
Ours:
  rg       : Retention-Guided Squeeze (adaptive step size + annealed threshold)
"""
import math
import torch


class Static:
    def __init__(self, name, start_rank, events):
        self.name, self._start, self.events = name, start_rank, dict(events)
        self.log = []

    def start_rank(self):
        return self._start

    def decide(self, step, spectra_fn):
        r = self.events.get(step)
        if r is not None:
            self.log.append((step, r))
        return r


def _halving(r_src, r_tgt):
    ranks, r = [r_src], r_src
    while r > r_tgt:
        r = max(r_tgt, r // 2)
        ranks.append(r)
    return ranks


def _stage_steps(ranks, total, min_steps=0):
    n = len(ranks)
    floor = min(min_steps, total // n)
    rest = total - floor * n
    w = [r / sum(ranks) for r in ranks]
    steps = [floor + int(rest * wi) for wi in w]
    steps[-1] += total - sum(steps)
    return steps


def _static_in_squeeze(name, total, r_src, r_tgt, min_steps):
    ranks = _halving(r_src, r_tgt)
    steps = _stage_steps(ranks, total, min_steps)
    events, t = {}, 0
    for r, s in zip(ranks[:-1], steps[:-1]):
        t += s
        events[t] = ranks[ranks.index(r) + 1]
    return Static(name, r_src, events)


class RetentionGuided:
    """Ours. At every `check_every` steps (after `dwell` steps since the last squeeze) pick the
    smallest rank r' >= max(r_tgt, ceil(r/max_shrink)) such that the q-quantile over modules of
    the retained spectral energy  V_m(r') = sum_{i<=r'} s_i^2 / sum_i s_i^2  is >= tau(t).
    tau(t) anneals linearly to 0 at t_slack = t_end - n_needed*dwell, so compression is
    guaranteed to reach r_tgt by t_end (the rest of the budget is fine-tuning at r_tgt).
    """

    def __init__(self, total, r_src, r_tgt, tau0=0.98, quantile=0.25, max_shrink=4,
                 dwell=200, check_every=25, end_frac=0.7):
        self.name = "rg"
        self.total, self.r_src, self.r_tgt = total, r_src, r_tgt
        self.tau0, self.q, self.max_shrink = tau0, quantile, max_shrink
        self.check_every = check_every
        self.t_end = int(end_frac * total)
        self.n_needed = max(1, math.ceil(math.log(max(r_src / r_tgt, 1.0001)) / math.log(max_shrink)))
        self.dwell = max(1, min(dwell, self.t_end // (self.n_needed + 1)))
        self.t_slack = max(1, self.t_end - self.n_needed * self.dwell)
        self.rank, self.last = r_src, 0
        self.log = []   # (step, old_rank, new_rank, tau, retention_at_choice)

    def start_rank(self):
        return self.r_src

    def tau(self, t):
        return self.tau0 * max(0.0, 1.0 - t / self.t_slack)

    def decide(self, step, spectra_fn):
        if self.rank <= self.r_tgt or step == 0 or step % self.check_every:
            return None
        if step - self.last < self.dwell or step > self.t_end + self.dwell:
            return None
        lo = max(self.r_tgt, math.ceil(self.rank / self.max_shrink))
        tau = self.tau(step)
        spectra = spectra_fn()
        energy = torch.stack([torch.cumsum(s ** 2, 0) / s.pow(2).sum().clamp_min(1e-30) for s in spectra])  # (modules, r)
        chosen, ret = None, None
        for r_new in range(lo, self.rank):                    # smallest admissible first
            ret_r = torch.quantile(energy[:, r_new - 1], self.q).item()
            if ret_r >= tau:
                chosen, ret = r_new, ret_r
                break
        if chosen is None:
            return None
        self.log.append((step, self.rank, chosen, tau, ret))
        self.rank, self.last = chosen, step
        return chosen


def make_policy(name, total, r_src, r_tgt, min_steps=200, cont_steps=200, **kw):
    if name == "direct":
        return Static("direct", r_tgt, {})
    if name == "post":
        return Static("post", r_src, {total: r_tgt})          # squeezed after the last step
    if name == "cont":
        return Static("cont", r_src, {total - cont_steps: r_tgt})
    if name == "in_std":
        return _static_in_squeeze("in_std", total, r_src, r_tgt, 0)
    if name == "in_min":
        return _static_in_squeeze("in_min", total, r_src, r_tgt, min_steps)
    if name == "rg":
        return RetentionGuided(total, r_src, r_tgt, **kw)
    raise ValueError(name)
