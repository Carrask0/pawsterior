import torch
import torch.nn as nn  # type: ignore
import torch.nn.functional as F

# ---------- time prior sampler ----------
def sample_time_prior(batch_size: int, device, alpha: float):
    """
    If alpha is None -> Uniform(0,1).
    Else sample t with pdf p(t) ∝ t^alpha on [0,1].
    Inverse-CDF: t = U^{1/(alpha+1)}, valid for alpha > -1.
    """
    if alpha is None:
        return torch.rand(batch_size, device=device)
    eps = 1e-6
    if alpha <= -1.0:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    u = torch.rand(batch_size, device=device).clamp(min=eps, max=1.0 - eps)
    return u.pow(1.0 / (alpha + 1.0))


# ---------- support constraint ----------

import torch
import torch.nn.functional as F

def parse_theta_spec(theta_spec: list[dict]) -> tuple[list[dict], bool]:
    blocks = []
    offset = 0
    has_cat = False
    for b in theta_spec:
        btype = b["type"].lower()
        if btype == "categorical_onehot":
            K = int(b["n_classes"])
            blocks.append({"type": "categorical_onehot", "sl": slice(offset, offset + K), "n_classes": K})
            offset += K
            has_cat = True
        elif btype == "continuous":
            d = int(b["dim"])
            blocks.append({"type": "continuous", "sl": slice(offset, offset + d), "dim": d})
            offset += d
        else:
            raise ValueError(f"Unknown theta_spec block type: {btype}")
    return blocks, has_cat


def _constrain_to_support(y: torch.Tensor, low, high) -> torch.Tensor:
    y_t = torch.tanh(y)

    def _as_row(v):
        if v is None:
            return None
        if isinstance(v, (float, int)):
            return torch.tensor([float(v)], device=y.device, dtype=y.dtype).view(1, -1)
        v_t = torch.as_tensor(v, device=y.device, dtype=y.dtype)
        if v_t.ndim == 0:
            v_t = v_t.view(1)
        return v_t.view(1, -1)

    low_t = _as_row(low)
    high_t = _as_row(high)
    if low_t is None and high_t is None:
        return y

    # one-sided: (0.0, None) or (None, 1.0)
    if high_t is None:
        # map to [low, +inf): use softplus + low
        return F.softplus(y) + low_t
    if low_t is None:
        # map to (-inf, high]: high - softplus
        return high_t - F.softplus(y)

    # two-sided
    return ((y_t + 1.0) * 0.5) * (high_t - low_t) + low_t


def map_to_manifold(
    y: torch.Tensor,
    *,
    mode: str,
    support_bounds=None,
    blocks: list[dict] | None = None,
    has_categorical: bool = False,
    temp: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    mode:
      - "loss": continuous -> support mapped, categorical -> logits (unchanged)
      - "sample": continuous -> support mapped, categorical -> softmax(logits/temp)
    """
    if support_bounds is None and not has_categorical:
        return y
    out = y.clone()

    # Continuous mapping
    if support_bounds is not None:
        low, high = support_bounds
        if blocks is None:
            out = _constrain_to_support(out, low, high)
        else:
            for b in blocks:
                if b["type"] == "continuous":
                    sl = b["sl"]
                    out[:, sl] = _constrain_to_support(out[:, sl], low, high)

    # Categorical mapping
    if has_categorical and blocks is not None:
        for b in blocks:
            if b["type"] != "categorical_onehot":
                continue
            sl = b["sl"]
            if mode == "loss":
                continue  # logits for CE
            elif mode == "sample":
                logits = out[:, sl]
                logits = logits - logits.max(dim=1, keepdim=True).values
                out[:, sl] = F.softmax(logits / max(float(temp), eps), dim=1)
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return out


def harden_categoricals_argmax(theta: torch.Tensor, *, blocks: list[dict] | None, has_categorical: bool) -> torch.Tensor:
    if not has_categorical or blocks is None:
        return theta
    out = theta.clone()
    for b in blocks:
        if b["type"] != "categorical_onehot":
            continue
        sl = b["sl"]
        K = b["n_classes"]
        cls = torch.argmax(out[:, sl], dim=1)
        out[:, sl] = F.one_hot(cls, num_classes=K).float()
    return out

# ============================================================================
# Model Architectures
# ============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (no NumPy, matches input dtype)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] (can be fp32 or bf16 under autocast)
        device = t.device
        dtype = t.dtype  # match autocast dtype (e.g., bfloat16)

        half_dim = self.dim // 2
        log_10000 = torch.log(torch.tensor(10000.0, device=device, dtype=dtype))
        exponent = torch.arange(half_dim, device=device, dtype=dtype)
        inv_freq = torch.exp(-log_10000 * exponent / (half_dim - 1))
        emb = t.view(-1, 1) * inv_freq.view(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
        return emb
    
####
# FOR LOSS
####


def blockwise_mse_ce(
    pred_loss_view: torch.Tensor,
    target_theta: torch.Tensor,
    *,
    blocks: list[dict] | None,
    has_categorical: bool,
    device: torch.device,
):
    # No blocks: pure MSE
    if blocks is None or not has_categorical:
        loss = F.mse_loss(pred_loss_view, target_theta)
        return loss, {"mse": float(loss.detach().cpu()), "ce": 0.0}

    mse_terms, ce_terms = [], []
    for b in blocks:
        sl = b["sl"]
        if b["type"] == "continuous":
            mse_terms.append(F.mse_loss(pred_loss_view[:, sl], target_theta[:, sl]))
        elif b["type"] == "categorical_onehot":
            targets = torch.argmax(target_theta[:, sl], dim=1)
            ce_terms.append(F.cross_entropy(pred_loss_view[:, sl], targets))
        else:
            raise RuntimeError("Unexpected block type")

    loss_mse = torch.stack(mse_terms).mean() if mse_terms else torch.tensor(0.0, device=device)
    loss_ce = torch.stack(ce_terms).mean() if ce_terms else torch.tensor(0.0, device=device)
    return loss_mse + loss_ce, {
        "mse": float(loss_mse.detach().cpu()),
        "ce": float(loss_ce.detach().cpu()),
    }



# ---------- fully-connected ResNet blocks ----------
class FCResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # PreNorm residual block (stabilizes deep MLPs)
        h = self.fc1(self.norm1(x))
        h = self.act(h)
        h = self.fc2(self.norm2(h))
        return x + h


class FCResNet(nn.Module):
    """
    Stem -> N residual blocks -> head
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[FCResBlock(hidden_dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        return h
