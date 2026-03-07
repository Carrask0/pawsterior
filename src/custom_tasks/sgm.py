import math
import torch
from torch.distributions import Categorical


# ----------------------------
# Prior: theta = onehot(z_path) only
# ----------------------------
class OneHotMarkovZPrior:
    """
    Samples theta = onehot(z_0..z_{T-1}) flattened, where:
      z_0 ~ Cat(pi0) or Uniform if pi0 is None
      z_t | z_{t-1} ~ Cat(Pi[z_{t-1}, :])

    Returns float tensor [N, T*K].
    """

    def __init__(self, K: int, T: int, Pi: torch.Tensor, pi0: torch.Tensor = None, device="cpu"):
        self.K = int(K)
        self.T = int(T)  # length of z_path (number of transitions)
        self.device = torch.device(device)

        Pi = torch.as_tensor(Pi, dtype=torch.float32, device=self.device)
        assert Pi.shape == (self.K, self.K), f"Pi must be [K,K], got {Pi.shape}"
        Pi = Pi / Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self.Pi = Pi

        self.pi0 = torch.ones(self.K, device=self.device) / self.K if pi0 is None else pi0.to(self.device)
        self.cat0 = Categorical(probs=self.pi0)

    def sample(self, sample_shape):
        N = int(sample_shape[0])

        # z: [N, T]
        z = torch.empty((N, self.T), dtype=torch.long, device=self.device)
        z[:, 0] = self.cat0.sample((N,))

        for t in range(1, self.T):
            prev = z[:, t - 1]
            zt = torch.empty((N,), dtype=torch.long, device=self.device)
            for k in range(self.K):
                idx = (prev == k)
                if idx.any():
                    cat = Categorical(probs=self.Pi[k])
                    zt[idx] = cat.sample((idx.sum().item(),))
            z[:, t] = zt

        onehot = torch.nn.functional.one_hot(z, num_classes=self.K).float()  # [N, T, K]
        return onehot.reshape(N, self.T * self.K)  # [N, T*K]

import torch

def sample_rotation_matrix(d: int, generator=None, device="cpu", dtype=torch.float32):
    """
    Haar-like random rotation in R^d with det=+1 via QR decomposition.
    Returns Q with Q^T Q = I and det(Q)=+1.
    """
    M = torch.randn((d, d), generator=generator, device=device, dtype=dtype)
    # QR decomposition
    Q, R = torch.linalg.qr(M)
    # Make Q uniform-ish on O(d): fix sign ambiguity using diag(R)
    diag = torch.sign(torch.diag(R))
    diag[diag == 0] = 1.0
    Q = Q @ torch.diag(diag)

    # Ensure det(Q)=+1 (proper rotation): if det=-1, flip one column
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def sample_distinct_rotations(K: int, d: int, generator=None, device="cpu",
                              min_fro_dist: float = 1.0, max_tries: int = 5000):
    """
    Sample K rotation matrices in SO(d) with a simple rejection criterion
    to encourage distinctness: ||Q_i - Q_j||_F >= min_fro_dist.
    """
    As = []
    tries = 0
    while len(As) < K:
        tries += 1
        if tries > max_tries:
            # fall back: accept whatever we have + random rotations
            while len(As) < K:
                As.append(sample_rotation_matrix(d, generator=generator, device=device))
            break

        Q = sample_rotation_matrix(d, generator=generator, device=device)

        ok = True
        for A in As:
            dist = torch.linalg.norm(Q - A)  # Frobenius norm
            if dist < min_fro_dist:
                ok = False
                break
        if ok:
            As.append(Q)

    return torch.stack(As, dim=0)  # [K, d, d]



# ----------------------------
# Task: Switching Linear Gaussian, theta = z_path only
# ----------------------------
class SwitchingGaussianMixture:
    """
    Time-series switching linear Gaussian model:

      z_0 ~ Cat(pi0) (default uniform)
      z_t | z_{t-1} ~ Cat(Pi[z_{t-1}, :])           for t=1..T-1

      x_0 ~ N(0, I)
      x_{t+1} | x_t, z_t=k ~ N(A_k x_t, sigma_k^2 I) for t=0..T-1

    Observation returned by simulator: x_seq = (x_0,...,x_T).
    Parameter theta: z_path = (z_0,...,z_{T-1}) only.

    sbibm-like subset:
      - get_prior_dist()
      - get_simulator()

    Plus:
      - sample_reference_posterior_theta(x_seq): exact posterior samples of z_path via FFBS
    """

    def __init__(
        self,
        d_x: int,
        K: int,
        T: int,               # z length = T, x length = T+1
        seed: int = 0,
        device: str = "cpu",
        stable_A_scale: float = 0.8,
    ):
        self.d_x = int(d_x)
        self.K = int(K)
        self.T = int(T)
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(int(seed))

        # sigma: [K]    
        sigma = torch.linspace(0.25, 0.6, steps=self.K)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device)
        assert self.sigma.shape == (self.K,)

        # Pi: [K,K]
        Pi = torch.full((self.K, self.K), 1.0 / self.K)
        Pi = 0.3 * Pi + 0.7 * torch.eye(self.K)  # sticky
        Pi = torch.as_tensor(Pi, dtype=torch.float32, device=self.device)
        Pi = Pi / Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self.Pi = Pi

        # pi0: [K]
        self.pi0 = torch.ones(self.K, device=self.device) / self.K # uniform

        # A_k: [K, d_x, d_x]
        A = sample_distinct_rotations(
            K=self.K,
            d=self.d_x,
            generator=g,
            device=self.device,
            min_fro_dist=1.8,   # encourage distinctness
        )
        A = stable_A_scale * A
        self.A = A.to(self.device)

        # I add this to make the task easier, otherwise i was getting C2ST of 0.99
        self.b = 2.0 * torch.randn(self.K, self.d_x, generator=g, device=self.device)  
        # anisotropic covariance for x0: diag with spread
        s0 = torch.linspace(0.3, 2.0, steps=self.d_x).to(self.device)
        self.x0_scale = s0  # store

        # Noisy observations
        self.obs_sigma = 0.5

        self.prior = OneHotMarkovZPrior(K=self.K, T=self.T, Pi=self.Pi, pi0=self.pi0, device=self.device)

    def get_prior_dist(self):
        return self.prior

    def _parse_theta(self, theta: torch.Tensor):
        """
        theta: [N, T*K] or [T*K]
        Returns:
          z_path: [N, T] long
          squeeze_out: bool
        """
        if theta.dim() == 1:
            theta_ = theta.unsqueeze(0)
            squeeze_out = True
        else:
            theta_ = theta
            squeeze_out = False

        N = theta_.shape[0]
        z_onehot = theta_.view(N, self.T, self.K)
        z_path = z_onehot.argmax(dim=-1).long()
        return z_path, squeeze_out

    def get_simulator(self):
        def sim(theta: torch.Tensor):
            """
            Simulate x trajectory given z_path.
            Samples x0 internally from N(0, I).

            Returns:
              x_seq: [N, T+1, d_x] or [T+1, d_x] if theta was 1D
            """
            z_path, squeeze_out = self._parse_theta(theta)
            N = z_path.shape[0]

            x = torch.empty((N, self.T + 1, self.d_x), dtype=torch.float32, device=self.device)
            x[:, 0, :] = torch.randn((N, self.d_x), device=self.device) * self.x0_scale


            for t in range(self.T):
                zt = z_path[:, t]      # [N]
                xt = x[:, t, :]        # [N, d_x]
                xt1 = torch.empty_like(xt)

                for k in range(self.K):
                    idx = (zt == k)
                    if idx.any():
                        mean = xt[idx] @ self.A[k].T + self.b[k]
                        noise = torch.randn_like(mean) * float(self.sigma[k])
                        xt1[idx] = mean + noise

                x[:, t + 1, :] = xt1

            # ---- observation noise  ----
            # y = x + torch.randn_like(x) * self.obs_sigma
            # return y.squeeze(0) if squeeze_out else y
            return x.squeeze(0) if squeeze_out else x
        return sim

    # ---------- Exact posterior over z_path given x_seq (FFBS) ----------

    def _log_trans_density(self, x_next: torch.Tensor, x_cur: torch.Tensor, k: int):
        """
        log p(x_{t+1} | x_t, z_t=k) with N(A_k x_t, sigma_k^2 I).
        x_next, x_cur: [d_x] or [N, d_x]
        returns: [N] logprob
        """
        s = float(self.sigma[k])
        s2 = s * s
        diff = x_next - (x_cur @ self.A[k].T + self.b[k])
        quad = (diff * diff).sum(dim=-1) / s2
        const = self.d_x * math.log(2.0 * math.pi * s2)
        return -0.5 * (quad + const)

    def sample_reference_posterior_theta(self, x_seq: torch.Tensor, num_samples: int = 1000):
        """
        Exact posterior sampling of theta=z_path given full observed x_seq (length T+1).
        FFBS.

        Args:
          x_seq: [T+1, d_x] or [1, T+1, d_x]
        Returns:
          z_samples: [num_samples, T] long
          theta_samples: [num_samples, T*K] float (onehot(z_path) flattened)
        """
        if x_seq.dim() == 3:
            assert x_seq.shape[0] == 1, "Batching not implemented; pass a single trajectory."
            x_seq = x_seq.squeeze(0)

        assert x_seq.shape == (self.T + 1, self.d_x), f"Expected {(self.T+1, self.d_x)}, got {tuple(x_seq.shape)}"
        x_seq = x_seq.to(self.device)

        # ll[t,k] = log p(x_{t+1} | x_t, z_t=k)
        ll = torch.empty((self.T, self.K), dtype=torch.float32, device=self.device)
        for t in range(self.T):
            x_cur = x_seq[t].unsqueeze(0)
            x_nxt = x_seq[t + 1].unsqueeze(0)
            for k in range(self.K):
                ll[t, k] = self._log_trans_density(x_nxt, x_cur, k)[0]

        log_Pi = torch.log(self.Pi.clamp_min(1e-30))
        log_pi0 = torch.log(self.pi0.clamp_min(1e-30))

        # forward: log_alpha[t,k] ∝ p(z_t=k | x_0:T)
        log_alpha = torch.empty((self.T, self.K), dtype=torch.float32, device=self.device)

        log_alpha[0] = log_pi0 + ll[0]
        log_alpha[0] -= torch.logsumexp(log_alpha[0], dim=0)

        for t in range(1, self.T):
            prev = log_alpha[t - 1].unsqueeze(1) + log_Pi  # [K,K]
            log_alpha[t] = ll[t] + torch.logsumexp(prev, dim=0)
            log_alpha[t] -= torch.logsumexp(log_alpha[t], dim=0)

        # backward sampling
        z_samples = torch.empty((num_samples, self.T), dtype=torch.long, device=self.device)

        probs_last = torch.softmax(log_alpha[self.T - 1], dim=0)
        z_samples[:, self.T - 1] = Categorical(probs=probs_last).sample((num_samples,))

        for t in range(self.T - 2, -1, -1):
            next_z = z_samples[:, t + 1]  # [N]
            logp = log_alpha[t].unsqueeze(0).repeat(num_samples, 1)  # [N,K]
            logp = logp + log_Pi[:, next_z].T                        # add log Pi[j, next_z[i]]
            probs = torch.softmax(logp, dim=1)
            z_samples[:, t] = torch.multinomial(probs, num_samples=1).squeeze(1)

        theta_samples = torch.nn.functional.one_hot(z_samples, num_classes=self.K).float()
        theta_samples = theta_samples.reshape(num_samples, self.T * self.K)  # [S, T*K]
        return z_samples, theta_samples
