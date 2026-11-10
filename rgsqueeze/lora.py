"""LoRA layer whose rank can be changed *during* training via exact SVD truncation.

Notation: dW = scale * B @ A, A in R^{r x in}, B in R^{out x r}.
Squeeze (memory-efficient form of LoRA-Squeeze, Alg. 2 in Vulic et al. 2026):
    B = Q_B R_B,  A^T = Q_A R_A      (thin QR, both r-dimensional)
    dW = Q_B (R_B R_A^T) Q_A^T = Q_B M Q_A^T,   M in R^{r x r}
    M = U S V^T  ->  singular values of dW are exactly S (Q_B, Q_A orthonormal)
    rank-k:  B' = Q_B U_k S_k^{1/2},  A' = S_k^{1/2} V_k^T Q_A^T
The dense out x in matrix is never formed. We use the exact SVD of the small r x r core
(cheap: r <= 128), not randomised SVD.
"""
import math
import torch
import torch.nn as nn


def _core_svd(A: torch.Tensor, B: torch.Tensor):
    A32, B32 = A.detach().float(), B.detach().float()
    QB, RB = torch.linalg.qr(B32)            # (out,r), (r,r)
    QA, RA = torch.linalg.qr(A32.t())        # (in,r),  (r,r)
    M = RB @ RA.t()
    U, S, Vh = torch.linalg.svd(M)
    return QB, QA, U, S, Vh


class ReRankLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float = 1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = scale
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    def forward(self, x):
        out = self.base(x)
        delta = (x.to(self.A.dtype) @ self.A.t()) @ self.B.t()
        return out + (self.scale * delta).to(out.dtype)

    @torch.no_grad()
    def spectrum(self) -> torch.Tensor:
        """Singular values of dW/scale, descending (length = current rank)."""
        return _core_svd(self.A, self.B)[3]

    @torch.no_grad()
    def squeeze(self, new_rank: int) -> torch.Tensor:
        """Truncate to new_rank (<= current rank). Returns the full pre-truncation spectrum."""
        QB, QA, U, S, Vh = _core_svd(self.A, self.B)
        k = max(1, min(int(new_rank), self.rank))
        root = S[:k].sqrt()
        newB = QB @ (U[:, :k] * root)                 # (out,k)
        newA = (root[:, None] * Vh[:k]) @ QA.t()      # (k,in)
        self.A = nn.Parameter(newA.to(self.A.dtype).contiguous())
        self.B = nn.Parameter(newB.to(self.B.dtype).contiguous())
        return S

    @torch.no_grad()
    def delta_w(self) -> torch.Tensor:
        return self.scale * (self.B.float() @ self.A.float())


def inject_lora(model: nn.Module, rank: int, target_suffixes, scale: float = 1.0):
    """Replace every nn.Linear whose name ends with one of target_suffixes."""
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and any(n.endswith(s) for s in target_suffixes)]
    for n in names:
        parent_name, _, child = n.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, ReRankLoRALinear(getattr(parent, child), rank, scale))
    return len(names)


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, ReRankLoRALinear)]


def lora_parameters(model):
    return [p for m in lora_modules(model) for p in (m.A, m.B)]


def current_rank(model) -> int:
    return lora_modules(model)[0].rank


@torch.no_grad()
def squeeze_all(model, new_rank: int):
    """Uniform rank for every module (keeps serving homogeneous, as LoRA-Squeeze intends)."""
    return [m.squeeze(new_rank) for m in lora_modules(model)]


@torch.no_grad()
def all_spectra(model):
    return [m.spectrum() for m in lora_modules(model)]
