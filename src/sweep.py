from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from filelock import FileLock


from .experiment import train_eval_once
from .paths import ckpt_dir
from .utils import ensure_dir, c2st_safe


# =============================================================================
# Sweep grid
# =============================================================================
def build_grid(cfg: dict) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """
    Build a Cartesian hyperparameter grid from cfg["grid"].

    Expected cfg structure (example):
    -------------------------------
    grid:
      hidden_dim: [512, 1024]
      num_blocks: [15, 16]
      learning_rate: [1e-3, 1e-4]
      alpha: [-0.5, 0, 1, 4]

    Returns
    -------
    keys: list[str]
        Ordered hyperparameter names.
    grid: list[tuple]
        List of tuples (hidden_dim, num_blocks, learning_rate, alpha), in deterministic order.
        The `grid_index` corresponds to the index into this list.
    """
    g = cfg["grid"]
    keys = ["hidden_dim", "num_blocks", "learning_rate", "alpha"]
    vals: Sequence[Sequence[Any]] = [g[k] for k in keys]
    grid = list(itertools.product(*vals))
    return keys, grid


# =============================================================================
# Best bookkeeping
# =============================================================================
def best_dir(model: str, init_dist: str, task: str, n_train: int) -> Path:
    """Directory containing the sweep's best.json and lock file."""
    return Path("artifacts") / "best" / model / init_dist / task / f"n{n_train}"


def best_path(model: str, init_dist: str, task: str, n_train: int) -> Path:
    """Path to best.json for a particular sweep setup."""
    return best_dir(model, init_dist, task, n_train) / "best.json"


def lock_path(model: str, init_dist: str, task: str, n_train: int) -> Path:
    """Path to lock file used to serialize updates to best.json."""
    return best_dir(model, init_dist, task, n_train) / ".lock"


def score_key(res: Dict[str, Any]) -> Tuple[float, float, int]:
    """
    Convert a result dictionary to a sortable key (lower is better).

    Parameters
    ----------
    res:
        Result dict returned by train_eval_once(...), expected to contain at least
        c2st_mean, optionally c2st_std and grid_index.

    Returns
    -------
    tuple
        Sort key for comparisons.
    """
    c2st_mean = float(res["c2st_mean"])
    c2st_std = float(res.get("c2st_std", 1e9))
    grid_index = int(res.get("grid_index", 10**9))
    return (abs(c2st_mean - 0.5), c2st_std, grid_index)


def load_best(model: str, init_dist: str, task: str, n_train: int) -> Dict[str, Any] | None:
    """Load best.json if present, else return None."""
    p = best_path(model, init_dist, task, n_train)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def save_best(best: Dict[str, Any], model: str, init_dist: str, task: str, n_train: int) -> None:
    """
    Atomically write best.json.

    We write to a temporary file then use os.replace, so readers never see a partial file.
    """
    bdir = best_dir(model, init_dist, task, n_train)
    ensure_dir(bdir)
    tmp = bdir / "best.json.tmp"
    with open(tmp, "w") as f:
        json.dump(best, f, indent=2)
    os.replace(tmp, best_path(model, init_dist, task, n_train))


def delete_run_artifacts(model: str, init_dist: str, task: str, n_train: int, grid_index: int | None) -> None:
    """
    Delete checkpoint folder for a specific grid run.

    This is called for:
    - any non-winning run immediately
    - the previous winner when a new best is found
    """
    d = ckpt_dir(model, init_dist, task, n_train, grid_index)
    shutil.rmtree(d, ignore_errors=True)


def maybe_update_best(cfg: dict, res: Dict[str, Any], grid_index: int) -> bool:
    """
    Compare this run's result to the current best and update if improved.

    Returns
    -------
    bool
        True if this run became the new best, False otherwise.
    """
    task = cfg["run"]["task_name"]
    model = cfg["run"]["model"]
    init_dist = cfg["run"]["init_dist"]
    n_train = int(cfg["run"]["n_train"])

    ensure_dir(best_dir(model, init_dist, task, n_train))
    lock = FileLock(str(lock_path(model, init_dist, task, n_train)))

    with lock:
        current = load_best(model, init_dist, task, n_train)

        if current is None or score_key(res) < score_key(current):
            # New winner: delete old winner's checkpoint (if it exists and differs)
            if current is not None:
                old_grid = current.get("grid_index", None)
                if old_grid is not None and int(old_grid) != int(grid_index):
                    delete_run_artifacts(model, init_dist, task, n_train, int(old_grid))

            save_best(res, model, init_dist, task, n_train)
            print(f"[BEST] Updated best: key={score_key(res)} grid={grid_index}")
            return True

        # Not best: delete this run immediately
        delete_run_artifacts(model, init_dist, task, n_train, int(grid_index))
        print(f"[DROP] Not best: key={score_key(res)} grid={grid_index}")
        return False


# =============================================================================
# Public: run one sweep index (streaming)
# =============================================================================
def run_sweep_index_streaming(cfg: dict, grid_index: int) -> Dict[str, Any]:
    """
    Train + evaluate one hyperparameter configuration and keep it only if it improves the best.

    Parameters
    ----------
    cfg:
        Parsed YAML configuration for the sweep. Must contain:
          - cfg["run"]: experiment definition (task_name, model, init_dist, n_train, ...)
          - cfg["grid"]: sweep lists for hidden_dim / num_blocks / learning_rate / alpha
    grid_index:
        Index into the Cartesian grid produced by build_grid(cfg).

    Returns
    -------
    dict
        The result dict produced by train_eval_once(...) for this run.

    """
    keys, grid = build_grid(cfg)
    if not (0 <= grid_index < len(grid)):
        raise ValueError(f"grid_index {grid_index} out of range 0..{len(grid) - 1}")

    values = grid[grid_index]

    # Create a shallow copy and override params for this grid point
    cfg = dict(cfg)
    cfg["params"] = dict(cfg.get("params", {}))
    for k, v in zip(keys, values):
        cfg["params"][k] = v

    # Train + evaluate
    res = train_eval_once(cfg, grid_index=grid_index)
    print(f"Completed training for grid index {grid_index} with result: {res}")
    res["grid_index"] = int(grid_index)  # guarantee it exists for best bookkeeping

    # Keep only if improved
    maybe_update_best(cfg, res, grid_index=int(grid_index))
    return res