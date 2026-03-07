import torch
import numpy as np
import os
import bz2
import re
import pandas as pd
from .models.fm import FlowMatchingVelocity,ResNetVelocityNet
from .models.vfm import FlowMatchingX0X1, ResNetX0X1Net


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def squeeze_if_batched_channel_first(x: torch.Tensor) -> torch.Tensor:
    """
    Some sbibm simulators (e.g. two_moons) can give [N,1,D].
    We want [N,D].
    """
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)  # [N,D]
    return x


def to_cpu_float_tensor(x):
    """
    Take numpy or tensor, return torch.float32 on CPU, squeezed to [N,D].
    Used for C2ST.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float().cpu()
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    return x


def load_reference_observation(task_files_root, task_name, obs_id):
    """
    Loads x_obs for a given benchmark observation id.
    Expects:
        {task_files_root}/{task_name}/num_observation_{obs_id}/observation.csv

    sbibm's observation.csv usually looks like:
        data_1,data_2,...,data_D
        0.123,0.456,...,...

    We:
      - read with pandas
      - take the first (and only) row of numeric values
      - return a CPU torch tensor of shape [x_dim] (float32)
    """
    obs_path = os.path.join(
        task_files_root,
        task_name,
        f"num_observation_{obs_id}",
        "observation.csv",
    )

    df = pd.read_csv(obs_path)  # header row with column names like data_1,...
    # df shape: [1, x_dim]
    x_obs_np = df.to_numpy(dtype=np.float32).squeeze(0)  # [x_dim]
    x_obs_t = torch.from_numpy(x_obs_np)  # CPU float32 tensor [x_dim]
    return x_obs_t


def load_reference_posterior(task_files_root, task_name, obs_id):
    """
    Loads the reference posterior θ ~ p(θ | x_obs) for obs_id.
    Expects:
        {task_files_root}/{task_name}/num_observation_{obs_id}/reference_posterior_samples.csv.bz2

    sbibm's reference_posterior_samples.csv.bz2 usually looks like:
        param_1,param_2,...,param_D
        0.12,   1.04,   ...
        0.11,   0.98,   ...
        ...

    We:
      - read with pandas (can handle bz2 via open)
      - convert the whole DataFrame to float32 numpy
      - return CPU torch tensor [N_ref, theta_dim]
    """
    post_path = os.path.join(
        task_files_root,
        task_name,
        f"num_observation_{obs_id}",
        "reference_posterior_samples.csv.bz2",
    )

    with bz2.open(post_path, "rb") as f:
        df = pd.read_csv(f)  # header row param_1,param_2,...
    post_np = df.to_numpy(dtype=np.float32)  # [N_ref, theta_dim]
    post_t = torch.from_numpy(post_np)  # CPU float32 tensor
    return post_t


def _parse_hparams_from_ckpt_name(path: str):
    """
    Parse hidden_dim, num_blocks, batch_size, lr, alpha from a checkpoint filename.
    Expected pattern (order matters):
      ..._resnet_hd{HD}_nb{NB}_bs{BS}_lr{LR}_a{ALPHA}.pth
    Returns a dict with keys: hidden_dim, num_blocks, batch_size, lr, alpha
    """
    # Accepts numbers like 2e-04, 1e-3, 0.0002, and alpha like -0.5, 0, 4, None
    pat = re.compile(
        r"_resnet_hd(?P<hd>\d+)_nb(?P<nb>\d+)_bs(?P<bs>\d+)_lr(?P<lr>[-+]?[\deE\.\-]+)_a(?P<a>[-+]?[\w\.\-]+)\.pth$"
    )
    m = pat.search(path)
    if not m:
        raise ValueError(
            f"Could not parse hyperparams from checkpoint name: {path}\n"
            "Expected suffix like: _resnet_hd{HD}_nb{NB}_bs{BS}_lr{LR}_a{ALPHA}.pth"
        )
    hd = int(m.group("hd"))
    nb = int(m.group("nb"))
    bs = int(m.group("bs"))
    lr_str = m.group("lr")
    a_str = m.group("a")

    # Convert lr
    try:
        lr = float(lr_str)
    except ValueError:
        lr = None

    # Convert alpha
    alpha = None if a_str.lower() in ("none", "nan") else float(a_str)

    return {
        "hidden_dim": hd,
        "num_blocks": nb,
        "batch_size": bs,
        "lr": lr,
        "alpha": alpha,
    }

def load_models_from_checkpoints(theta_dim, x_dim, device, config, task_name, n_samples, init_dist, grid_index, theta_prior=None):
    from paths import ckpt_path as _ckpt_path
    import torch

    # Map natural task to base task cfg for dims/spec
    cfg_task_name = "switching_gaussian_linear" if task_name == "sgl_natural" else task_name
    task_cfg = config["tasks"][cfg_task_name]
    theta_spec = task_cfg.get("theta_spec", None)

    support_bounds = None
    if task_cfg.get("prior_dist", "").lower() == "uniform" and "support" in task_cfg:
        support_bounds = (task_cfg["support"][0], task_cfg["support"][1])

    # Build nets using *the current grid config* 
    hidden_dim = config["training"]["hidden_dim"]
    num_blocks = config["training"]["num_blocks"]

    vel_net = ResNetVelocityNet(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, num_blocks=num_blocks)
    it_net  = ResNetX0X1Net(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, num_blocks=num_blocks)

    fm_model = FlowMatchingVelocity(vel_net, device=device, init_dist=init_dist, theta_prior=theta_prior)
    it_model = FlowMatchingX0X1(it_net, support_bounds=support_bounds, device=device, theta_spec=theta_spec, init_dist=init_dist, theta_prior=theta_prior)

    fm_ckpt_path = _ckpt_path("velocity", init_dist, task_name, int(n_samples), grid_index)
    it_ckpt_path = _ckpt_path("x0x1", init_dist, task_name, int(n_samples), grid_index)

    fm_state = torch.load(str(fm_ckpt_path), map_location="cpu", weights_only=True)
    it_state = torch.load(str(it_ckpt_path), map_location="cpu", weights_only=True)

    fm_model.prediction_net.load_state_dict(fm_state["state_dict"], strict=True)
    it_model.prediction_net.load_state_dict(it_state["state_dict"], strict=True)

    fm_model.prediction_net.to(device).eval()
    it_model.prediction_net.to(device).eval()

    return fm_model, it_model
