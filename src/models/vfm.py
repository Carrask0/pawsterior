import torch
# torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision("high")
import torch.nn as nn  # type: ignore
import torch.nn.functional as F
from .common import sample_time_prior, map_to_manifold, SinusoidalTimeEmbedding, FCResNet, blockwise_mse_ce, parse_theta_spec, harden_categoricals_argmax


# DEBUG
def _chk(name, z):
    ok = torch.isfinite(z).all().item()
    if not ok:
        print(f"[NaN] {name}: finite={ok} shape={tuple(z.shape)} dtype={z.dtype}")
        # show a few offending values
        bad = (~torch.isfinite(z)).nonzero(as_tuple=False)[:10]
        print("[NaN] first bad idx:", bad.tolist())
        raise ValueError(f"{name} contains NaN/Inf")


# ---------- ResNet X0/X1 heads ----------
class ResNetX0X1Net(nn.Module):
    def __init__(
        self,
        theta_dim,
        x_dim,
        hidden_dim=1024,
        num_blocks=16,
        time_in_dim=64,
        time_out_dim=256,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_in_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_out_dim),
        )
        in_dim = theta_dim + x_dim + time_out_dim
        self.backbone = FCResNet(in_dim, hidden_dim, num_blocks)
        self.head_x0 = nn.Linear(hidden_dim, theta_dim)
        self.head_x1 = nn.Linear(hidden_dim, theta_dim)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def forward(self, theta, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        h = torch.cat([theta, x, t_emb], dim=1)
        h = self.backbone(h)
        return self.head_x0(h), self.head_x1(h)


# ============================================================================
#  Models
# ============================================================================

class FlowMatchingX0X1:
    def __init__(self, prediction_net, support_bounds, *, init_dist="gaussian", alpha=0.0, theta_prior=None, device=None, theta_spec=None, beta=0.2, simplex_temp=1.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.prediction_net = prediction_net.to(self.device)

        self.support_bounds = support_bounds
        self.init_dist = init_dist
        self.alpha = alpha
        self.theta_prior = theta_prior
        self.beta = float(beta)
        self.simplex_temp = float(simplex_temp)

        self.blocks = None
        self.has_categorical = False
        if theta_spec is not None:
            self.blocks, self.has_categorical = parse_theta_spec(theta_spec)

    def sample_initial_theta(self, B, D):
        if self.init_dist == "gaussian":
            return torch.randn(B, D, device=self.device)
        if self.init_dist == "theta_prior":
            return self.theta_prior.sample((B,)).to(self.device).float()
        raise ValueError(f"Unknown init_dist: {self.init_dist}")

    def compute_loss(self, theta_1: torch.Tensor, x_obs: torch.Tensor, *, alpha: float | None = None):
        B, D = theta_1.shape
        theta_0 = self.sample_initial_theta(B, D)
        t = sample_time_prior(B, device=self.device, alpha=self.alpha if alpha is None else alpha)
        theta_t = (1 - t.view(-1, 1)) * theta_0 + t.view(-1, 1) * theta_1

        x0_raw, x1_raw = self.prediction_net(theta_t, x_obs, t)

        # mapping policy you want:
        x1_loss_view = map_to_manifold(
            x1_raw, mode="loss", support_bounds=self.support_bounds,
            blocks=self.blocks, has_categorical=self.has_categorical, temp=self.simplex_temp
        )
        if self.init_dist == "theta_prior":
            x0_loss_view = map_to_manifold(
                x0_raw, mode="loss", support_bounds=self.support_bounds,
                blocks=self.blocks, has_categorical=self.has_categorical, temp=self.simplex_temp
            )
        else:
            x0_loss_view = x0_raw

        loss_x0, m0 = blockwise_mse_ce(x0_loss_view, theta_0, blocks=self.blocks, has_categorical=self.has_categorical, device=self.device)
        loss_x1, m1 = blockwise_mse_ce(x1_loss_view, theta_1, blocks=self.blocks, has_categorical=self.has_categorical, device=self.device)

        loss = self.beta * loss_x0 + (1.0 - self.beta) * loss_x1
        return loss, {
            "loss": float(loss.detach().cpu()),
            "loss_x0": float(loss_x0.detach().cpu()),
            "loss_x1": float(loss_x1.detach().cpu()),
            "loss_x0_mse": m0["mse"], "loss_x0_ce": m0["ce"],
            "loss_x1_mse": m1["mse"], "loss_x1_ce": m1["ce"],
        }

    @torch.no_grad()
    def sample(self, num_samples, theta_dim, x_obs, num_steps=100, temp=1.0, hard_final=True):
        theta = self.sample_initial_theta(num_samples, theta_dim)
        dt = 1.0 / num_steps

        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if x_obs.shape[0] == 1:
            x_obs = x_obs.repeat(num_samples, 1)

        ts = torch.linspace(0, 1 - 1/num_steps, steps=num_steps, device=self.device)
        for i in range(num_steps):
            t = ts[i].expand(num_samples)
            x0_raw, x1_raw = self.prediction_net(theta, x_obs, t)

            x1_state = map_to_manifold(
                x1_raw, mode="sample", support_bounds=self.support_bounds,
                blocks=self.blocks, has_categorical=self.has_categorical, temp=temp
            )
            if self.init_dist == "theta_prior":
                x0_state = map_to_manifold(
                    x0_raw, mode="sample", support_bounds=self.support_bounds,
                    blocks=self.blocks, has_categorical=self.has_categorical, temp=temp
                )
            else:
                x0_state = x0_raw

            theta = theta + (x1_state - x0_state) * dt

        theta = map_to_manifold(
            theta, mode="sample", support_bounds=self.support_bounds,
            blocks=self.blocks, has_categorical=self.has_categorical, temp=temp
        )
        if hard_final:
            theta = harden_categoricals_argmax(theta, blocks=self.blocks, has_categorical=self.has_categorical)
        return theta
    

class FlowMatchingX1:
    def __init__(self, prediction_net, support_bounds, *, init_dist="gaussian", alpha=0.0, theta_prior=None, device=None, theta_spec=None, denom_clamp=0.05, simplex_temp=1.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.prediction_net = prediction_net.to(self.device)
        self.support_bounds = support_bounds
        self.init_dist = init_dist
        self.alpha = alpha
        self.theta_prior = theta_prior
        self.denom_clamp = float(denom_clamp)
        self.simplex_temp = float(simplex_temp)

        self.blocks = None
        self.has_categorical = False
        if theta_spec is not None:
            self.blocks, self.has_categorical = parse_theta_spec(theta_spec)

    def sample_initial_theta(self, B, D):
        if self.init_dist == "gaussian":
            return torch.randn(B, D, device=self.device)
        if self.init_dist == "theta_prior":
            return self.theta_prior.sample((B,)).to(self.device).float()
        raise ValueError(f"Unknown init_dist: {self.init_dist}")

    def compute_loss(self, theta_1: torch.Tensor, x_obs: torch.Tensor, *, alpha: float | None = None):
        B, D = theta_1.shape
        theta_0 = self.sample_initial_theta(B, D)
        t = sample_time_prior(B, device=self.device, alpha=self.alpha if alpha is None else alpha)
        theta_t = (1 - t.view(-1, 1)) * theta_0 + t.view(-1, 1) * theta_1

        x1_raw = self.prediction_net(theta_t, x_obs, t)
        x1_loss_view = map_to_manifold(
            x1_raw, mode="loss", support_bounds=self.support_bounds,
            blocks=self.blocks, has_categorical=self.has_categorical, temp=self.simplex_temp
        )
        loss, m = blockwise_mse_ce(x1_loss_view, theta_1, blocks=self.blocks, has_categorical=self.has_categorical, device=self.device)
        return loss, {"loss": float(loss.detach().cpu()), "loss_mse": m["mse"], "loss_ce": m["ce"]}

    @torch.no_grad()
    def sample(self, num_samples, theta_dim, x_obs, num_steps=100, temp=1.0, hard_final=True):
        theta = self.sample_initial_theta(num_samples, theta_dim)
        dt = 1.0 / num_steps

        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if x_obs.shape[0] == 1:
            x_obs = x_obs.repeat(num_samples, 1)

        ts = torch.linspace(0, 1 - 1/num_steps, steps=num_steps, device=self.device)
        for i in range(num_steps):
            t = ts[i].expand(num_samples)
            x1_raw = self.prediction_net(theta, x_obs, t)
            x1_state = map_to_manifold(
                x1_raw, mode="sample", support_bounds=self.support_bounds,
                blocks=self.blocks, has_categorical=self.has_categorical, temp=temp
            )
            denom = (1.0 - t).clamp_min(self.denom_clamp).view(-1, 1)
            v = (x1_state - theta) / denom
            theta = theta + v * dt

        theta = map_to_manifold(
            theta, mode="sample", support_bounds=self.support_bounds,
            blocks=self.blocks, has_categorical=self.has_categorical, temp=temp
        )
        if hard_final:
            theta = harden_categoricals_argmax(theta, blocks=self.blocks, has_categorical=self.has_categorical)
        return theta

