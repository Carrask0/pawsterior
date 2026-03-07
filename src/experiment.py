from __future__ import annotations
import json
import numpy as np
import torch

from .tasks import get_task_and_cfg
from .paths import ckpt_path, ckpt_meta_path, results_path
from .utils import set_seed, ensure_dir, to_jsonable, c2st_safe
from .helpers import load_reference_observation, load_reference_posterior, to_cpu_float_tensor

from .models.fm import ResNetVelocityNet, FlowMatchingVelocity
from .models.vfm import ResNetX0X1Net, FlowMatchingX0X1, FlowMatchingX1
from .models.train import train_model


def build_model(cfg: dict, task_cfg: dict, theta_prior, device, theta_spec=None):
    model_name = cfg["run"]["model"]
    p = cfg["params"]

    task_name = cfg["run"]["task_name"]
    if task_name == "sgm":
        # For SGM, we need to override theta_dim and x_dim based on the task config
        dim_parameters = int(task_cfg["T"])  * int(task_cfg["K"])  # T*K parameters
        dim_data = int(task_cfg["d_x"]) * (int(task_cfg["T"]) + 1)  # (T+1)*d_x data dimension
    else:
        dim_parameters = int(task_cfg["dim_parameters"])
        dim_data = int(task_cfg["dim_data"])

    theta_dim = dim_parameters
    x_dim = dim_data
    theta_spec = theta_spec 
    alpha = float(cfg["params"]["alpha"])

    support_bounds = None
    if task_cfg.get("prior_dist", "").lower() == "uniform" and "support" in task_cfg:
        lo, hi = task_cfg["support"]
        support_bounds = (float(lo), float(hi))

    if model_name == "velocity":
        net = ResNetVelocityNet(theta_dim=theta_dim, x_dim=x_dim,
                                hidden_dim=int(p["hidden_dim"]), num_blocks=int(p["num_blocks"]))
        return FlowMatchingVelocity(net, device=device,
                                   init_dist=cfg["run"]["init_dist"], alpha=alpha, theta_prior=theta_prior), theta_dim, x_dim, task_cfg
    elif model_name == "x0x1":
        net = ResNetX0X1Net(theta_dim=theta_dim, x_dim=x_dim,
                            hidden_dim=int(p["hidden_dim"]), num_blocks=int(p["num_blocks"]))
        return FlowMatchingX0X1(net, support_bounds=support_bounds, theta_spec=theta_spec, alpha=alpha,
                                device=device, init_dist=cfg["run"]["init_dist"], theta_prior=theta_prior), theta_dim, x_dim, task_cfg
    elif model_name == "x1":
        net = ResNetVelocityNet(
            theta_dim=theta_dim, x_dim=x_dim,
            hidden_dim=int(p["hidden_dim"]), num_blocks=int(p["num_blocks"])
        )
        return (
            FlowMatchingX1(
                net,
                support_bounds=support_bounds,
                init_dist=cfg["run"]["init_dist"],
                theta_prior=theta_prior,
                alpha=alpha,
                device=device,
                theta_spec=theta_spec,
                denom_clamp=0.05,          # or expose via cfg
                simplex_temp=1.0,          # or expose via cfg
            ),
            theta_dim,
            x_dim,
            task_cfg,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_eval_once(cfg: dict, *, grid_index: int | None = None):
    set_seed(int(cfg.get("seed", 0)))

    task_name = cfg["run"]["task_name"]
    task, task_cfg, theta_spec = get_task_and_cfg(task_name, cfg)
    theta_prior = task.get_prior_dist()
    simulator = task.get_simulator()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    n_train = int(cfg["run"]["n_train"])
    n_val = int(cfg["run"]["n_val"])
    total = n_train + n_val

    # --- data
    theta_all = theta_prior.sample((total,)).float()
    x_all = simulator(theta_all).float()  

    if x_all.dim() == 3:
        x_all = x_all.reshape(x_all.shape[0], -1)
    theta_tr, theta_va = theta_all[:n_train], theta_all[n_train:]
    x_tr, x_va = x_all[:n_train], x_all[n_train:]

    # --- model
    model, theta_dim, x_dim, task_cfg = build_model(cfg, task_cfg, theta_prior, device, theta_spec=theta_spec)

    # --- train
    hist = train_model(
        model,
        model_type=cfg["run"]["model"],
        theta_train=theta_tr, x_train=x_tr,
        theta_val=theta_va, x_val=x_va,
        config=cfg,
        task_name=task_name,
        n_samples=n_train,
        model_key=cfg["run"]["model"],
        init_dist=cfg["run"]["init_dist"],
        grid_index=grid_index,
    )

    # --- eval
    res = eval_once(cfg, task_name, task_cfg, model, theta_dim, device, grid_index=grid_index)

    # --- persist result json
    out = results_path(cfg["run"]["model"], cfg["run"]["init_dist"], task_name, n_train, (grid_index or 0))
    ensure_dir(out.parent)
    with open(out, "w") as f:
        json.dump(to_jsonable(res), f, indent=2)

    return res

def eval_once(cfg: dict, task_name: str, task_cfg: dict, model, theta_dim: int, device, *, grid_index: int | None):
    n_post = int(cfg["run"]["n_posterior_samples"])
    n_obs = int(cfg["run"]["n_obs_eval"])

    c2sts = []
    for obs_id in range(1, n_obs + 1):
        x_obs = load_reference_observation("task_files", task_name, obs_id).to(device).float()
        ref_post = load_reference_posterior("task_files", task_name, obs_id)

        with torch.no_grad():
            hyp = model.sample(num_samples=n_post, theta_dim=theta_dim, x_obs=x_obs)

        ref = to_cpu_float_tensor(ref_post)[:n_post]
        hyp = to_cpu_float_tensor(hyp.detach().cpu().numpy())[:n_post]

        val = float(c2st_safe(ref, hyp))
        c2sts.append(val)

    return {
        "task": task_name,
        "model": cfg["run"]["model"],
        "init_dist": cfg["run"]["init_dist"],
        "n_train": int(cfg["run"]["n_train"]),
        "grid_index": grid_index,
        "params": dict(cfg["params"]),
        "c2st_scores": c2sts,
        "c2st_mean": float(np.mean(c2sts)),
        "c2st_std": float(np.std(c2sts)),
    }