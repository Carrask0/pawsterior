# create_task_files_sgm.py
#
# Generates sbibm-style task_files for the time-series Switching Gaussian Model (SGM),
# where:
#   - theta = z_path only (one-hot flattened): shape [T*K]
#   - observation = x_0..x_T flattened: shape [(T+1)*d_x]
#   - reference posterior samples = exact FFBS samples of z_path given x_seq
#
# Output: pawsterior/task_files/sgm/num_observation_i/{observation.csv,true_parameters.csv,...}

import os
import bz2
import numpy as np
import pandas as pd
import torch

from sgm import SwitchingGaussianMixture


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_bz2_csv(df: pd.DataFrame, path: str):
    """Write a DataFrame to a bz2-compressed CSV with header."""
    with bz2.open(path, "wt") as f:
        df.to_csv(f, index=False)


def main(
    root="task_files",
    task_name="sgm",
    num_obs=10,
    num_ref=1000,          # reference posterior samples per observation
    seed=1000020,            # starting seed
    cfg=None,                # should match your config.yaml
):
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "task_files")

    # ---- MUST match config.yaml ----
    if cfg is None:
        cfg = dict(
            d_x=5,
            K=10,
            T=10,
            task_seed=0,
            stable_A_scale=1.0,  # For rotations, use 1.0
        )

    # Task instance 
    task = SwitchingGaussianMixture(
        d_x=int(cfg["d_x"]),
        K=int(cfg["K"]),
        T=int(cfg["T"]),
        seed=int(cfg["task_seed"]),
        device="cpu",  
    )

    prior = task.get_prior_dist()
    sim = task.get_simulator()

    # Dimensions
    K = int(task.K)
    T = int(task.T)
    d_x = int(task.d_x)

    theta_dim = T * K
    obs_dim = (T + 1) * d_x

    base_seed = int(seed)

    out_task_dir = os.path.join(root, task_name)
    ensure_dir(out_task_dir)

    # (Optional) Print the internally constructed parameters 
    # print("Internal sigma:", task.sigma)
    # print("Internal Pi:", task.Pi)
    # print("Internal pi0:", task.pi0)

    for obs_id in range(1, num_obs + 1):
        obs_dir = os.path.join(out_task_dir, f"num_observation_{obs_id}")
        ensure_dir(obs_dir)

        obs_seed = base_seed + obs_id
        torch.manual_seed(obs_seed)
        np.random.seed(obs_seed)

        # ---- sample true theta and generate observation ----
        theta_true = prior.sample((1,)).squeeze(0).float()  # [theta_dim]
        x_seq = sim(theta_true).float()                     # [T+1, d_x] (or [1,T+1,d_x])

        if x_seq.dim() == 3 and x_seq.shape[0] == 1:
            x_seq = x_seq.squeeze(0)

        x_obs = x_seq.reshape(-1).contiguous()              # [obs_dim]

        assert theta_true.numel() == theta_dim, (theta_true.shape, theta_dim)
        assert x_obs.numel() == obs_dim, (x_obs.shape, obs_dim)

        # ---- reference posterior samples (exact) ----
        # returns (z_samples [N,T], theta_samples [N, theta_dim])
        z_samps, ref_theta = task.sample_reference_posterior_theta(x_seq, num_samples=num_ref)
        ref_theta = ref_theta.float()

        # ---- Write observation_seed.csv ----
        df_seed = pd.DataFrame(
            {"observation_seed": [obs_seed], "num_observation": [obs_id]}
        )
        df_seed.to_csv(os.path.join(obs_dir, "observation_seed.csv"), index=False)

        # ---- Write observation.csv with data_1,...,data_obs_dim ----
        obs_cols = [f"data_{i}" for i in range(1, obs_dim + 1)]
        df_obs = pd.DataFrame([x_obs.cpu().numpy().astype(np.float32)], columns=obs_cols)
        df_obs.to_csv(os.path.join(obs_dir, "observation.csv"), index=False)

        # ---- Write true_parameters.csv with parameter_1,...,parameter_theta_dim ----
        theta_cols = [f"parameter_{i}" for i in range(1, theta_dim + 1)]
        df_theta = pd.DataFrame([theta_true.cpu().numpy().astype(np.float32)], columns=theta_cols)
        df_theta.to_csv(os.path.join(obs_dir, "true_parameters.csv"), index=False)

        # ---- Write reference_posterior_samples.csv.bz2 ----
        df_ref = pd.DataFrame(ref_theta.cpu().numpy().astype(np.float32), columns=theta_cols)
        write_bz2_csv(df_ref, os.path.join(obs_dir, "reference_posterior_samples.csv.bz2"))

        print(
            f"[OK] obs_id={obs_id} seed={obs_seed} "
            f"x_seq_shape={tuple(x_seq.shape)} x_obs_shape={tuple(x_obs.shape)} "
            f"theta_true_shape={tuple(theta_true.shape)} ref_shape={tuple(ref_theta.shape)} "
            f"-> {obs_dir}"
        )


if __name__ == "__main__":
    main()
