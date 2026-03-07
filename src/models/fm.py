import torch
# torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision("high")
import torch.nn as nn  # type: ignore
import torch.nn.functional as F
from .common import sample_time_prior, SinusoidalTimeEmbedding, FCResNet



# ---------- ResNet Velocity head ----------
class ResNetVelocityNet(nn.Module):
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
        self.head = nn.Linear(hidden_dim, theta_dim)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def forward(self, theta, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        h = torch.cat([theta, x, t_emb], dim=1)
        h = self.backbone(h)
        return self.head(h)


    
class FlowMatchingVelocity:
    """Standard flow matching with velocity prediction."""

    def __init__(self, prediction_net, init_dist="gaussian", alpha=0.0, theta_prior=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.prediction_net = prediction_net.to(self.device)  
        self.init_dist = init_dist
        self.alpha = alpha
        self.theta_prior = theta_prior

    def sample_initial_theta(self, num_samples, theta_dim):
        if self.init_dist == "gaussian":
            theta = torch.randn(num_samples, theta_dim, device=self.device)
        elif self.init_dist == "theta_prior":
            if self.theta_prior is None:
                raise ValueError("init_dist='theta_prior' requires theta_prior to be provided.")
            theta = self.theta_prior.sample((num_samples,)).to(self.device).float()
        else:
            raise ValueError(f"Unknown init_dist: {self.init_dist}")
        return theta
    

    def compute_loss(self, theta_1, x_obs):
        """
        theta_1: [B, Dθ]
        x_obs:   [B, Dx]
        """
        B = theta_1.shape[0]
        theta_0 = self.sample_initial_theta(B, theta_1.shape[1])
        alpha = self.alpha
        t = sample_time_prior(B, device=self.device, alpha=alpha)  # [B]
        t_expand = t.view(-1, 1)
        theta_t = (1 - t_expand) * theta_0 + t_expand * theta_1

        v_pred = self.prediction_net(theta_t, x_obs, t)
        u_t = theta_1 - theta_0
        mse = F.mse_loss(v_pred, u_t)
        return mse, {"loss_velocity": mse.item()}

    # Euler sampling
    @torch.no_grad()
    def sample(self, num_samples, theta_dim, x_obs, num_steps=100):
        """
        x_obs: tensor shape [Dx] or [1,Dx] or [num_samples,Dx].
               Will be repeated to match num_samples if needed.
        """
        # Sampling from initial distribution
        theta = self.sample_initial_theta(num_samples, theta_dim)

        dt = 1.0 / num_steps

        # Make x_obs batch-sized
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)  # [1,Dx]
        if x_obs.shape[0] == 1:
            x_obs = x_obs.repeat(num_samples, 1)  # [B,Dx]

        ts = torch.linspace(0, 1 - 1/num_steps, steps=num_steps, device=self.device)
        for i in range(num_steps):
            t = ts[i].expand(num_samples)  # [B]
            v = self.prediction_net(theta, x_obs, t)
            theta = theta + v * dt

        return theta