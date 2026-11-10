import math, sys, torch, torch.nn as nn
sys.path.insert(0, ".")
from rgsqueeze import ReRankLoRALinear, make_policy, squeeze_all, inject_lora, lora_modules
from rgsqueeze.lora import all_spectra

torch.manual_seed(0)

def make_layer(r=16, din=48, dout=40):
    m = ReRankLoRALinear(nn.Linear(din, dout), r)
    with torch.no_grad():
        m.B.normal_(0, 0.3)
        m.A.mul_(2.0)
    return m

def test_spectrum_matches_dense_svd():
    m = make_layer()
    s = m.spectrum()
    ref = torch.linalg.svdvals(m.delta_w())
    assert torch.allclose(s, ref[: len(s)], atol=1e-4), (s[:3], ref[:3])

def test_full_rank_squeeze_is_lossless_and_forward_equal():
    m = make_layer()
    x = torch.randn(7, 48)
    y0, w0 = m(x), m.delta_w()
    m.squeeze(16)
    assert (m.delta_w() - w0).abs().max() < 1e-4
    assert (m(x) - y0).abs().max() < 1e-4

def test_truncation_is_optimal_low_rank_approx():
    m = make_layer()
    w0 = m.delta_w()
    S = torch.linalg.svdvals(w0)
    for k in (1, 4, 9):
        m2 = make_layer(); m2.load_state_dict(m.state_dict())
        m2.squeeze(k)
        err = torch.linalg.norm(m2.delta_w() - w0)
        assert abs(err - S[k:].pow(2).sum().sqrt()) < 1e-4          # Eckart-Young
        assert m2.rank == k

def test_zero_B_does_not_crash():
    m = ReRankLoRALinear(nn.Linear(10, 10), 4)
    m.squeeze(2)
    assert m.rank == 2 and torch.isfinite(m.A).all() and torch.isfinite(m.B).all()

def test_static_schedules_cover_budget():
    for name in ("in_std", "in_min"):
        p = make_policy(name, 3000, 128, 1)
        ev = sorted(p.events.items())
        assert [r for _, r in ev] == [64, 32, 16, 8, 4, 2, 1]
        assert all(0 < t < 3000 for t, _ in ev)
    pin = make_policy("in_min", 3000, 128, 1)
    pstd = make_policy("in_std", 3000, 128, 1)
    assert min(pin.events) >= 200            # rank-128 stage gets at least the floor
    # in_min gives the late (low-rank) stages more steps than in_std
    assert sorted(pin.events)[-1] - sorted(pin.events)[-2] > sorted(pstd.events)[-1] - sorted(pstd.events)[-2]

def test_rg_always_reaches_target_and_is_monotone():
    for r_tgt in (1, 4, 8):
        for flat in (True, False):
            T = 2000
            pol = make_policy("rg", T, 64, r_tgt)
            # fake spectra: flat (worst case, nothing satisfies tau) or steep
            s = torch.ones(64) if flat else torch.tensor([0.5 ** i for i in range(64)])
            rank, ranks = pol.start_rank(), []
            for t in range(T + 1):
                r = pol.decide(t, lambda: [s[:rank] for _ in range(6)])
                if r is not None:
                    assert r < rank and r >= r_tgt
                    rank = r; ranks.append(r)
            assert rank == r_tgt, (r_tgt, flat, ranks)
            assert pol.log[-1][0] <= int(0.7 * T) + pol.dwell

def test_rg_steeper_spectrum_squeezes_earlier_and_further():
    T = 2000
    out = {}
    for kind in ("steep", "flat"):
        pol = make_policy("rg", T, 64, 4)
        s = (torch.tensor([0.5 ** i for i in range(64)]) if kind == "steep" else torch.arange(64, 0, -1).float() ** 0.3)
        rank = 64
        for t in range(T + 1):
            r = pol.decide(t, lambda: [s[:rank] for _ in range(6)])
            if r: rank = r
        out[kind] = pol.log[0]
    assert out["steep"][0] <= out["flat"][0]            # first squeeze no later
    assert out["steep"][2] <= out["flat"][2]            # and at least as aggressive

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
