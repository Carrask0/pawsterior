from __future__ import annotations
import sbibm
import yaml

from .custom_tasks.sgm import SwitchingGaussianMixture

def get_task_and_cfg(task_name: str, cfg: dict):
    task_cfg_dir = "configs/tasks.yaml"
    with open(task_cfg_dir, "r") as f:
        tasks_cfg = yaml.safe_load(f)

    if task_name == "sgm": # SGM requires custom handling because of the temporal dimension
        task = SwitchingGaussianMixture(
            d_x=int(tasks_cfg[task_name]["d_x"]),
            K=int(tasks_cfg[task_name]["K"]),
            T=int(tasks_cfg[task_name]["T"]),
        )
        theta_spec = []
        for t in range(int(tasks_cfg[task_name]["T"])):
            theta_spec.append(
                dict(type="categorical_onehot", name=f"z_{t}", n_classes=int(tasks_cfg[task_name]["K"]))
            )
        cfg = dict(cfg)  # copy to avoid mutating global config

        return task, tasks_cfg[task_name], theta_spec

    # default sbibm
    return sbibm.get_task(task_name), tasks_cfg[task_name], None