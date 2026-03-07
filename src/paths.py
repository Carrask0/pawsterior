from __future__ import annotations
from pathlib import Path

ARTIFACTS_ROOT = Path("artifacts")

def task_folder(task_name: str) -> str:
    # keep task name as-is (including sgl_natural) for storage
    return task_name

def grid_folder(grid_index: int | None) -> str:
    if grid_index is None:
        return "grid_manual"
    return f"grid_{grid_index:03d}"

def ckpt_dir(model: str, init_dist: str, task_name: str, n_samples: int, grid_index: int | None) -> Path:
    return (ARTIFACTS_ROOT / "checkpoints" / model / init_dist / task_folder(task_name) / f"n{n_samples}" / grid_folder(grid_index))

def ckpt_path(model: str, init_dist: str, task_name: str, n_samples: int, grid_index: int | None) -> Path:
    return ckpt_dir(model, init_dist, task_name, n_samples, grid_index) / "ckpt.pth"

def ckpt_meta_path(model: str, init_dist: str, task_name: str, n_samples: int, grid_index: int | None) -> Path:
    return ckpt_dir(model, init_dist, task_name, n_samples, grid_index) / "meta.json"

def results_dir(model: str, init_dist: str, task_name: str, n_samples: int) -> Path:
    return ARTIFACTS_ROOT / "results" / model / init_dist / task_folder(task_name) / f"n{n_samples}"

def results_path(model: str, init_dist: str, task_name: str, n_samples: int, grid_index: int) -> Path:
    return results_dir(model, init_dist, task_name, n_samples) / f"grid_{grid_index:03d}.json"

def summary_path(model: str, init_dist: str, task_name: str, n_samples: int) -> Path:
    return results_dir(model, init_dist, task_name, n_samples) / "summary.csv"
