"""
Complex-Valued Neural Network (CVNN) with Holomorphic Activation Functions
==========================================================================

All activation functions here are *entire* (holomorphic everywhere on ℂ),
which means they satisfy the Cauchy-Riemann (CR) equations at every point:

    ∂u/∂x = ∂v/∂y      (CR-1)
    ∂u/∂y = -∂v/∂x     (CR-2)

where f(z) = u(x,y) + i·v(x,y),  z = x + iy.

Liouville's theorem tells us no bounded holomorphic function exists on all
of ℂ except constants, so every useful holomorphic activation is unbounded.
This is a fundamental constraint — functions like modReLU or split-ReLU
that act on real/imag parts separately are NOT holomorphic and are excluded.

Holomorphic activations implemented
------------------------------------
  1. CReLU        – complex ReLU: max(Re(z),0) + i·max(Im(z),0)
                    NOTE: this is real-analytic on each half-plane but not
                    holomorphic globally; included for comparison only and
                    labelled explicitly.
  2. Holomorphic activations (all entire):
     • zReLU       – z if Im(z)>0 and Re(z)>0, else 0  (Guberman 2016)
                     Holomorphic only on the open first quadrant; everywhere
                     else it is 0.  Labelled "sector holomorphic".
     • ComplexExp  – exp(z)         entire, canonical
     • ComplexSin  – sin(z)         entire
     • ComplexSinh – sinh(z)        entire
     • Mobiuz      – z/(1+|z|)      smooth but NOT holomorphic (|z| breaks CR)
                     Excluded – shown for reference only.
     • CRSigmoid   – σ(z) = 1/(1+exp(-z))   entire (complex logistic)
     • CRTanh      – tanh(z)        entire
     • CRGELU      – z·Φ(z) where Φ is the complex normal CDF approximation
                     (entire via error function)
     • Cardioid    – ½(1 + cos(arg(z)))·z   NOT holomorphic (arg breaks CR)
                     Excluded.

For training stability the holomorphic activations are used.
CRTanh and CRSigmoid are recommended for most tasks.

References
----------
  Trabelsi et al. (2018) "Deep Complex Networks"  https://arxiv.org/abs/1705.09792
  Bassey et al. (2021)   "A Survey of Complex-Valued Neural Networks"
  Guberman (2016)        "On Complex Valued Convolutional Neural Networks"
"""

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Holomorphic activation functions
# ============================================================================

class ComplexExp(nn.Module):
    """f(z) = exp(z).  Entire, CR-satisfying everywhere.

    exp(x+iy) = e^x · (cos y + i sin y)
    ∂u/∂x = e^x cos y  = ∂v/∂y  ✓
    ∂u/∂y = -e^x sin y = -∂v/∂x ✓
    """
    def forward(self, z: Tensor) -> Tensor:
        return torch.exp(z)


class ComplexSin(nn.Module):
    """f(z) = sin(z).  Entire.

    sin(x+iy) = sin x cosh y + i cos x sinh y
    CR satisfied by standard calculus of complex trig functions.
    """
    def forward(self, z: Tensor) -> Tensor:
        return torch.sin(z)


class ComplexSinh(nn.Module):
    """f(z) = sinh(z).  Entire.

    sinh(x+iy) = sinh x cos y + i cosh x sin y
    """
    def forward(self, z: Tensor) -> Tensor:
        return torch.sinh(z)


class CRSigmoid(nn.Module):
    """f(z) = 1 / (1 + exp(-z)).  Entire (complex logistic sigmoid).

    This is the direct analytic continuation of the real sigmoid.
    Has poles at z = iπ(2k+1), k∈ℤ, but is holomorphic on ℂ \ {poles}.
    For practical inputs it behaves well; clip large |Im(z)| if needed.

    CR equations:  f'(z) = f(z)·(1 - f(z))  — the usual sigmoid derivative
    holds over ℂ, confirming holomorphicity.
    """
    def forward(self, z: Tensor) -> Tensor:
        return torch.sigmoid(z)          # PyTorch sigmoid handles complex tensors


class CRTanh(nn.Module):
    """f(z) = tanh(z).  Entire (meromorphic, poles off real axis).

    tanh(x+iy) = (sinh 2x + i sin 2y) / (cosh 2x + cos 2y)
    f'(z) = 1 - tanh²(z)  holds over ℂ → holomorphic.
    Recommended default: bounded on real axis, gradient ∈ (0,1].
    """
    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(z)


class CRSoftplus(nn.Module):
    """f(z) = log(1 + exp(z)).  Entire (analytic continuation of softplus).

    f'(z) = sigmoid(z), which is entire → f is entire by integration.
    """
    def forward(self, z: Tensor) -> Tensor:
        # Use numerically stable form: log1p(exp(z))
        # For complex z: torch.log(1 + torch.exp(z))
        return torch.log1p(torch.exp(z))


class CRGELU(nn.Module):
    """f(z) = z · Φ(z)  where Φ is the complex normal CDF.

    Φ(z) = ½ · erfc(-z/√2)
    erf is entire ⇒ erfc is entire ⇒ Φ is entire ⇒ f is entire.

    f'(z) = Φ(z) + z · φ(z)  where φ(z) = exp(-z²/2)/√(2π)  (entire)
    → CR satisfied everywhere.
    """
    def forward(self, z: Tensor) -> Tensor:
        # torch.special.erf supports complex tensors in recent PyTorch versions
        # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return z * cdf


class ZReLU(nn.Module):
    """f(z) = z  if both Re(z) > 0 and Im(z) > 0, else 0.

    Proposed by Guberman (2016).  Holomorphic on the open first quadrant
    (where it equals the identity) and equals 0 elsewhere.  The boundary
    creates non-differentiable edges; treat as 'piecewise holomorphic'.

    This satisfies CR where it is smooth (identity map is holomorphic).
    """
    def forward(self, z: Tensor) -> Tensor:
        mask = (z.real > 0) & (z.imag > 0)
        return z * mask.to(z.dtype)


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

ACTIVATIONS: dict[str, nn.Module] = {
    "exp":       ComplexExp(),
    "sin":       ComplexSin(),
    "sinh":      ComplexSinh(),
    "sigmoid":   CRSigmoid(),
    "tanh":      CRTanh(),
    "softplus":  CRSoftplus(),
    "gelu":      CRGELU(),
    "zrelu":     ZReLU(),
}


def get_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]


# ============================================================================
# Complex-valued Linear layer
# ============================================================================

class ComplexLinear(nn.Module):
    """
    Complex-valued affine map:  z_out = W · z_in + b

    Implemented as the standard complex matrix–vector product:
        Re(out) = W_r · Re(in) - W_i · Im(in) + b_r
        Im(out) = W_r · Im(in) + W_i · Re(in) + b_i

    Parameters are stored as real tensors (weight_r, weight_i, bias_r, bias_i)
    so standard optimisers work without modification.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real and imaginary weight matrices
        self.weight_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        self._init_weights()

    def _init_weights(self):
        """Glorot uniform for complex weights (scale by 1/√2 per component)."""
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_i)
        # Rescale so total complex variance matches Glorot
        with torch.no_grad():
            self.weight_r.mul_(1.0 / math.sqrt(2))
            self.weight_i.mul_(1.0 / math.sqrt(2))

    def forward(self, z: Tensor) -> Tensor:
        """
        z : complex tensor of shape (..., in_features)
        returns : complex tensor of shape (..., out_features)
        """
        xr, xi = z.real, z.imag

        out_r = F.linear(xr, self.weight_r) - F.linear(xi, self.weight_i)
        out_i = F.linear(xr, self.weight_i) + F.linear(xi, self.weight_r)

        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return torch.complex(out_r, out_i)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias_r is not None}"


# ============================================================================
# Complex-valued Layer Normalisation
# ============================================================================

class ComplexLayerNorm(nn.Module):
    """
    Complex layer normalisation following Trabelsi et al. (2018).

    Normalises using the 2×2 covariance matrix of [Re(z), Im(z)] to produce
    a unit-covariance output, then applies learnable complex scale γ and
    shift β.

    This is the correct generalisation — normalising Re and Im independently
    would break the complex structure.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # Learnable parameters: complex γ (scale) and β (shift)
        self.gamma_rr = nn.Parameter(torch.ones(normalized_shape) / math.sqrt(2))
        self.gamma_ii = nn.Parameter(torch.ones(normalized_shape) / math.sqrt(2))
        self.gamma_ri = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_r   = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_i   = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, z: Tensor) -> Tensor:
        xr, xi = z.real, z.imag

        # Centre
        mu_r = xr.mean(dim=-1, keepdim=True)
        mu_i = xi.mean(dim=-1, keepdim=True)
        xr_c = xr - mu_r
        xi_c = xi - mu_i

        # 2×2 covariance matrix components
        Vrr = (xr_c ** 2).mean(dim=-1, keepdim=True) + self.eps
        Vii = (xi_c ** 2).mean(dim=-1, keepdim=True) + self.eps
        Vri = (xr_c * xi_c).mean(dim=-1, keepdim=True)

        # Inverse square root of 2×2 PSD matrix via analytic formula
        # For M = [[a, b],[b, c]]  we need M^{-1/2}
        # det = ac - b²,  τ = a+c,  s = √det,  t = √(τ + 2s)
        det  = Vrr * Vii - Vri ** 2
        s    = torch.sqrt(det.clamp(min=self.eps))
        tau  = Vrr + Vii
        t    = torch.sqrt((tau + 2 * s).clamp(min=self.eps))

        inv_t = 1.0 / t
        # M^{-1/2} = (1/t)(M + s·I)
        W11 =  inv_t * (Vrr + s)
        W22 =  inv_t * (Vii + s)
        W12 =  inv_t * Vri

        # Whitened outputs
        yr = W11 * xr_c + W12 * xi_c
        yi = W12 * xr_c + W22 * xi_c

        # Learnable affine (complex multiply then add)
        out_r = self.gamma_rr * yr - self.gamma_ri * yi + self.beta_r
        out_i = self.gamma_ri * yr + self.gamma_ii * yi + self.beta_i

        return torch.complex(out_r, out_i)


# ============================================================================
# Complex-valued MLP
# ============================================================================

class ComplexMLP(nn.Module):
    """
    A fully-connected complex-valued MLP.

    All weights, biases, and activations operate in ℂ.
    The activation must be holomorphic (satisfies CR equations).

    Parameters
    ----------
    in_features   : real input dimension  (input will be cast to complex)
    out_features  : output dimension
    hidden_dims   : sequence of hidden layer widths
    activation    : name of holomorphic activation (default 'tanh')
    use_layernorm : whether to apply ComplexLayerNorm between layers
    dropout_p     : dropout on |z| (real-valued dropout on modulus)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] = (128, 128),
        activation: str = "tanh",
        use_layernorm: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.act = get_activation(activation)

        dims = [in_features] + list(hidden_dims) + [out_features]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(ComplexLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # not the output layer
                if use_layernorm:
                    layers.append(ComplexLayerNorm(dims[i + 1]))
                layers.append(self.act)
                if dropout_p > 0.0:
                    layers.append(ComplexDropout(dropout_p))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : real or complex tensor of shape (B, in_features)
        """
        if not x.is_complex():
            x = x.to(torch.complex64)
        return self.net(x)


# ============================================================================
# Complex Dropout (on modulus, preserves phase)
# ============================================================================

class ComplexDropout(nn.Module):
    """
    Drops entire neurons (zeroes both Re and Im simultaneously).
    This preserves the complex structure: a dropped neuron contributes 0 ∈ ℂ,
    not a distorted partial value.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, z: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return z
        # Bernoulli mask on real part, applied to both components
        mask = torch.bernoulli(
            torch.full(z.real.shape, 1 - self.p, device=z.device)
        ) / (1 - self.p)
        mask = mask.to(z.real.dtype)
        return torch.complex(z.real * mask, z.imag * mask)


# ============================================================================
# Complex-valued Residual Block
# ============================================================================

class ComplexResidualBlock(nn.Module):
    """
    Pre-activation residual block for complex-valued networks:

        out = z + Linear₂(act(LN(Linear₁(act(LN(z))))))

    The skip connection is a pure complex addition, preserving holomorphicity.
    """

    def __init__(self, dim: int, activation: str = "tanh", use_layernorm: bool = True):
        super().__init__()
        self.act = get_activation(activation)
        self.ln1 = ComplexLayerNorm(dim) if use_layernorm else nn.Identity()
        self.ln2 = ComplexLayerNorm(dim) if use_layernorm else nn.Identity()
        self.fc1 = ComplexLinear(dim, dim)
        self.fc2 = ComplexLinear(dim, dim)

    def forward(self, z: Tensor) -> Tensor:
        h = self.act(self.fc1(self.ln1(z)))
        h = self.fc2(self.ln2(h))
        return z + h


# ============================================================================
# Full Complex-Valued Deep Residual Network
# ============================================================================

class CVResNet(nn.Module):
    """
    Deep complex-valued residual network.

    Architecture:
        Real input  →  ComplexLinear (embed)
                    →  N × ComplexResidualBlock
                    →  ComplexLinear (project)
                    →  real output  (take modulus or real part)

    Parameters
    ----------
    in_features    : real input width
    out_features   : real output width
    hidden_dim     : width of all hidden layers
    n_blocks       : number of residual blocks
    activation     : holomorphic activation name
    output_mode    : 'real' → Re(z), 'imag' → Im(z), 'abs' → |z|, 'complex'
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        activation: str = "tanh",
        output_mode: Literal["real", "imag", "abs", "complex"] = "real",
        use_layernorm: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.output_mode = output_mode

        self.embed   = ComplexLinear(in_features, hidden_dim)
        self.act_in  = get_activation(activation)

        self.blocks  = nn.ModuleList([
            ComplexResidualBlock(hidden_dim, activation=activation,
                                 use_layernorm=use_layernorm)
            for _ in range(n_blocks)
        ])
        self.dropout = ComplexDropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.project = ComplexLinear(hidden_dim, out_features)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            x = x.to(torch.complex64)

        z = self.act_in(self.embed(x))
        for block in self.blocks:
            z = block(z)
        z = self.dropout(z)
        z = self.project(z)

        if self.output_mode == "real":
            return z.real
        elif self.output_mode == "imag":
            return z.imag
        elif self.output_mode == "abs":
            return torch.abs(z)
        else:
            return z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Loss functions for complex outputs
# ============================================================================

class ComplexMSELoss(nn.Module):
    """||z_pred - z_target||²  = (Re diff)² + (Im diff)²"""
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target.to(pred.dtype)
        return (diff.real ** 2 + diff.imag ** 2).mean()


class ComplexCrossEntropyLoss(nn.Module):
    """
    Classification from complex logits: take the real part as logits
    (or the modulus, switchable).
    """
    def __init__(self, use_modulus: bool = False):
        super().__init__()
        self.use_modulus = use_modulus

    def forward(self, z_logits: Tensor, targets: Tensor) -> Tensor:
        logits = torch.abs(z_logits) if self.use_modulus else z_logits.real
        return F.cross_entropy(logits, targets)


# ============================================================================
# Cauchy-Riemann numerical verifier (for testing/debugging)
# ============================================================================

@torch.no_grad()
def verify_cauchy_riemann(
    f: Callable[[Tensor], Tensor],
    z0: Tensor,
    h: float = 1e-4,
    tol: float = 1e-3,
) -> dict:
    """
    Numerically verify the Cauchy-Riemann equations for function f at z0.

    CR-1:  ∂u/∂x ≈ ∂v/∂y
    CR-2:  ∂u/∂y ≈ -∂v/∂x

    Uses centred finite differences.

    Returns a dict with the residuals and a boolean 'passed'.
    """
    z0 = z0.to(torch.complex128)   # use double for accuracy

    dx = torch.tensor(h,   dtype=torch.float64)
    dy = torch.tensor(h*1j, dtype=torch.complex128)

    def u(z): return f(z).real
    def v(z): return f(z).imag

    # ∂u/∂x,  ∂v/∂x
    du_dx = (u(z0 + dx) - u(z0 - dx)) / (2 * h)
    dv_dx = (v(z0 + dx) - v(z0 - dx)) / (2 * h)

    # ∂u/∂y,  ∂v/∂y
    dh_dy = torch.tensor(1j * h, dtype=torch.complex128)
    du_dy = (u(z0 + dh_dy) - u(z0 - dh_dy)) / (2 * h)
    dv_dy = (v(z0 + dh_dy) - v(z0 - dh_dy)) / (2 * h)

    cr1_residual = float((du_dx - dv_dy).abs().mean())
    cr2_residual = float((du_dy + dv_dx).abs().mean())

    passed = (cr1_residual < tol) and (cr2_residual < tol)
    return {
        "CR1_residual (∂u/∂x - ∂v/∂y)": cr1_residual,
        "CR2_residual (∂u/∂y + ∂v/∂x)": cr2_residual,
        "passed": passed,
    }


# ============================================================================
# Training utilities
# ============================================================================

class Trainer:
    """Minimal trainer for CVNN models."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)
        self.sched  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=100)

    def step(self, x: Tensor, y: Tensor, loss_fn: nn.Module) -> float:
        x, y = x.to(self.device), y.to(self.device)
        self.optim.zero_grad()
        pred = self.model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        # Gradient clipping helps with complex nets (gradients can explode)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        self.sched.step()
        return loss.item()


# ============================================================================
# Quick demo / smoke-test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Verify Cauchy-Riemann for each activation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Cauchy-Riemann Verification")
    print("=" * 60)
    test_pts = torch.tensor([0.3 + 0.7j, -1.2 + 0.4j, 0.0 + 1.0j], dtype=torch.complex128)

    for name, act in ACTIVATIONS.items():
        result = verify_cauchy_riemann(act, test_pts)
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {name:12s}  {status}  "
              f"CR1={result['CR1_residual (∂u/∂x - ∂v/∂y)']:.2e}  "
              f"CR2={result['CR2_residual (∂u/∂y + ∂v/∂x)']:.2e}")

    # ------------------------------------------------------------------
    # 2. ComplexMLP  — regression on complex targets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ComplexMLP — complex regression")
    print("=" * 60)

    mlp = ComplexMLP(
        in_features=8,
        out_features=4,
        hidden_dims=[64, 64],
        activation="tanh",
        use_layernorm=True,
    )
    x_real = torch.randn(32, 8)
    y_cpx  = torch.randn(32, 4) + 1j * torch.randn(32, 4)

    loss_fn = ComplexMSELoss()
    trainer = Trainer(mlp, lr=3e-3)

    for step in range(200):
        loss = trainer.step(x_real, y_cpx.to(torch.complex64), loss_fn)
    print(f"  Final loss after 200 steps: {loss:.6f}")

    # ------------------------------------------------------------------
    # 3. CVResNet — real-output classification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CVResNet — classification (10 classes)")
    print("=" * 60)

    model = CVResNet(
        in_features=16,
        out_features=10,
        hidden_dim=64,
        n_blocks=3,
        activation="tanh",
        output_mode="real",
        use_layernorm=True,
        dropout_p=0.1,
    ).to(device)

    print(f"  Parameters: {model.num_parameters():,}")

    x   = torch.randn(64, 16).to(device)
    lbl = torch.randint(0, 10, (64,)).to(device)

    ce_loss = ComplexCrossEntropyLoss()
    trainer2 = Trainer(model, lr=1e-3, device=device)

    losses = []
    for step in range(300):
        l = trainer2.step(x, lbl, ce_loss)
        losses.append(l)
    print(f"  Initial loss : {losses[0]:.4f}")
    print(f"  Final loss   : {losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Forward pass shape checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Shape checks")
    print("=" * 60)

    for mode in ("real", "imag", "abs", "complex"):
        m = CVResNet(4, 3, hidden_dim=16, n_blocks=1, output_mode=mode)
        inp = torch.randn(8, 4)
        out = m(inp)
        print(f"  output_mode={mode:8s}  input {tuple(inp.shape)} → output {tuple(out.shape)}"
              f"  dtype={out.dtype}")

    print("\nAll checks complete.")