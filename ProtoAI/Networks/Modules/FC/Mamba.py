"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
Paper: https://arxiv.org/abs/2312.00752
Albert Gu, Tri Dao (2023)

This implementation covers:
  - S6 (selective scan) core
  - MambaBlock (single layer)
  - Mamba (full stacked model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Utility: 1-D convolution with a causal (left-padded) kernel
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Depthwise causal convolution used inside the SSM block."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            groups=channels,        # depthwise
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        return self.conv(x)[:, :, : -self.padding] if self.padding else self.conv(x)


# ---------------------------------------------------------------------------
# S6: the Selective State Space layer
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Selective Scan (S6) as described in the Mamba paper.

    Parameters
    ----------
    d_model : int
        Model (inner) dimension D.
    d_state : int
        SSM state size N  (paper default: 16).
    dt_rank : int | "auto"
        Rank of the Δ projection. "auto" → ceil(d_model / 16).
    dt_min / dt_max : float
        Bounds for the log-uniform initialisation of Δ.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input-dependent projections (Δ, B, C are all functions of x)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)

        # Δ: low-rank → full rank
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialise dt_proj so that softplus(dt_proj(·)) lies in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))          # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True                  # skip weight-init hooks

        # A: fixed S4D-real initialisation, stored as log for positivity
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: skip connection (scalar per channel)
        self.D = nn.Parameter(torch.ones(d_model))
        self.D._no_weight_decay = True

    # ------------------------------------------------------------------
    # Core selective scan (parallel prefix / associative scan in full
    # implementations; here we use the sequential form for clarity)
    # ------------------------------------------------------------------

    def _selective_scan(
        self,
        u: torch.Tensor,    # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,    # (D, N)
        B: torch.Tensor,    # (B, L, N)
        C: torch.Tensor,    # (B, L, N)
        D: torch.Tensor,    # (D,)
    ) -> torch.Tensor:
        B_sz, L, D = u.shape
        N = A.shape[1]

        # Discretise A and B with ZOH
        # delta: (B, L, D) → (B, L, D, 1); A: (D, N) → (1, 1, D, N)
        delta_A = torch.exp(delta.unsqueeze(-1) * A[None, None])  # (B, L, D, N)
        delta_B_u = (delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1))  # (B, L, D, N)

        # Sequential scan
        x = torch.zeros(B_sz, D, N, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            x = delta_A[:, i] * x + delta_B_u[:, i]           # (B, D, N)
            y = (x * C[:, i].unsqueeze(1)).sum(-1)             # (B, D)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + u * D[None, None]   # skip connection
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, D)
        returns : (B, L, D)
        """
        B, L, D = x.shape

        # Project to Δ (dt_rank), B (d_state), C (d_state)
        xz = self.x_proj(x)                                      # (B, L, dt_rank + 2*N)
        delta_raw, B_seq, C_seq = xz.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta_raw))               # (B, L, D)
        A = -torch.exp(self.A_log.float())                        # (D, N)

        return self._selective_scan(x, delta, A, B_seq, C_seq, self.D)


# ---------------------------------------------------------------------------
# MambaBlock: one complete Mamba layer
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Single Mamba block:

        x ──▶ expand ──▶ depthwise conv ──▶ SiLU ──▶ SSM ──▶ ⊗ ──▶ project ──▶ out
                                                              ↑
                                              x ──▶ expand ──▶ SiLU (gate)

    Parameters
    ----------
    d_model   : residual stream dimension
    d_state   : SSM state size  (default 16)
    d_conv    : depthwise conv kernel size (default 4)
    expand    : inner expansion factor  E  (default 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        # Input projection: split into two halves (one for SSM, one for gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Depthwise causal conv on the SSM branch
        self.conv1d = CausalConv1d(self.d_inner, kernel_size=d_conv)

        # SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, dt_rank=dt_rank)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Layer norm (applied before the block, i.e. pre-norm)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        returns : (B, L, d_model)   (residual NOT added here; caller handles it)
        """
        residual = x
        x = self.norm(x)

        # Split into SSM branch (z) and gate branch (g)
        xz = self.in_proj(x)                                  # (B, L, 2*d_inner)
        z, g = xz.chunk(2, dim=-1)                            # each (B, L, d_inner)

        # SSM branch: conv → activation → SSM
        z_conv = self.conv1d(z.transpose(1, 2)).transpose(1, 2)  # (B, L, d_inner)
        z_act = F.silu(z_conv)
        y = self.ssm(z_act)                                    # (B, L, d_inner)

        # Gating
        y = y * F.silu(g)

        return self.out_proj(y) + residual                     # residual added here


# ---------------------------------------------------------------------------
# Full Mamba model
# ---------------------------------------------------------------------------

class Mamba(nn.Module):
    """
    Stack of Mamba blocks for sequence modelling.

    Parameters
    ----------
    d_model    : model dimension
    n_layers   : number of MambaBlocks
    vocab_size : if > 0, adds an embedding layer + LM head
    d_state    : SSM state size
    d_conv     : depthwise conv width
    expand     : inner expansion factor
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        vocab_size: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        bias: bool = False,
        pad_vocab_size_multiple: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Optional embedding / LM head
        if vocab_size > 0:
            if vocab_size % pad_vocab_size_multiple != 0:
                vocab_size += pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Weight tying
            self.lm_head.weight = self.embedding.weight
        else:
            self.embedding = None
            self.lm_head = None

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                       dt_rank=dt_rank, bias=bias)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L)  token ids  — if vocab_size > 0
          | (B, L, d_model)    — if used as a backbone (no embedding)

        returns:
          (B, L, vocab_size)  logits  — if lm_head exists
          (B, L, d_model)     hidden  — otherwise
        """
        if self.embedding is not None:
            x = self.embedding(x)       # (B, L) → (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        if self.lm_head is not None:
            return self.lm_head(x)
        return x

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def num_parameters(self, exclude_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embedding and self.embedding is not None:
            total -= self.embedding.weight.numel()
        return total


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Language-model mode -----
    model = Mamba(
        d_model=256,
        n_layers=4,
        vocab_size=8192,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    print(f"Parameters (excl. embedding): {model.num_parameters():,}")

    tokens = torch.randint(0, 8192, (2, 128)).to(device)
    logits = model(tokens)
    print(f"Input shape  : {tokens.shape}")
    print(f"Output shape : {logits.shape}")   # (2, 128, 8192)

    # ----- Backbone mode (no vocab) -----
    backbone = Mamba(d_model=128, n_layers=2).to(device)
    feats = torch.randn(4, 64, 128).to(device)
    out = backbone(feats)
    print(f"\nBackbone output : {out.shape}")  # (4, 64, 128)
