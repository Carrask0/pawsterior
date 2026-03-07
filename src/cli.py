from __future__ import annotations

import argparse
import yaml
from typing import Any, Dict

from .experiment import train_eval_once
from .sweep import run_sweep_index_streaming


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _cast_scalar(v: str) -> Any:
    """Best-effort scalar casting for overrides."""
    v_strip = v.strip()
    low = v_strip.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v_strip)
    except ValueError:
        pass
    try:
        return float(v_strip)
    except ValueError:
        pass
    return v_strip


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """
    Apply overrides of the form:
        a.b.c=value

    Example:
        --override run.task_name=sir run.n_train=100000 params.learning_rate=1e-4
    """
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Invalid override '{kv}'. Expected key=value.")
        key, val = kv.split("=", 1)
        parts = key.split(".")
        cur: Dict[str, Any] = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _cast_scalar(val)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(prog="sbi-it")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # Single train + evaluate
    # -------------------------
    p_run = sub.add_parser("run", help="Train + evaluate once.")
    p_run.add_argument("--config", required=True, help="Path to a YAML config.")
    p_run.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides: key=value with dotted paths (e.g. run.task_name=sir).",
    )

    # -------------------------
    # Sweep: one grid index
    # -------------------------
    p_sw = sub.add_parser(
        "sweep-index",
        help="Run one hyperparameter configuration (train+eval) and keep only if best.",
    )
    p_sw.add_argument("--config", required=True, help="Path to a sweep YAML config.")
    p_sw.add_argument("--grid-index", type=int, required=True, help="Index into the sweep grid.")

    args = p.parse_args()

    if args.cmd == "run":
        cfg = load_yaml(args.config)
        cfg = apply_overrides(cfg, args.override)
        train_eval_once(cfg)

    elif args.cmd == "sweep-index":
        cfg = load_yaml(args.config)
        run_sweep_index_streaming(cfg, args.grid_index)


if __name__ == "__main__":
    main()