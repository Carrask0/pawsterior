from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


# =============================================================================
# Metric: safe C2ST (used by evaluation code)
# =============================================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
def c2st_safe(ref: torch.Tensor, hyp: torch.Tensor, seed: int = 0, n_splits: int = 5) -> float:
    """
    Classification two-sample test (C2ST) with a small MLP classifier

    Parameters
    ----------
    ref:
        Reference posterior samples, shape [N, D] (torch tensor).
    hyp:
        Model posterior samples, shape [N, D] (torch tensor).
    seed:
        Random seed for CV splits and classifier initialization.
    n_splits:
        Number of StratifiedKFold splits.

    Returns
    -------
    float
        Mean classification accuracy across folds.
    """
    X = torch.cat([ref, hyp], dim=0).detach().cpu().numpy().astype(np.float32)
    y = np.concatenate([np.zeros(len(ref)), np.ones(len(hyp))]).astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs: List[float] = []
    for tr_idx, te_idx in cv.split(X, y):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0  # safe standardization

        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd

        clf = MLPClassifier(
            hidden_layer_sizes=(50, 50),
            max_iter=200,
            random_state=seed,
            early_stopping=True,
        )
        clf.fit(Xtr_s, ytr)
        accs.append(float(clf.score(Xte_s, yte)))

    return float(np.mean(accs))



def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and (if available) PyTorch.

    Parameters
    ----------
    seed:
        Random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists (mkdir -p).

    Accepts either a directory path or a file path.
    If `path` has a suffix (e.g. ".json"), its parent directory is created.

    Parameters
    ----------
    path:
        Path to a directory or a file.

    Returns
    -------
    Path
        The directory that exists.
    """
    p = Path(path)
    # If looks like a file path, create its parent.
    d = p.parent if p.suffix else p
    d.mkdir(parents=True, exist_ok=True)
    return d


def to_jsonable(obj: Any) -> Any:
    """
    Convert common objects into JSON-serializable types.

    Handles:
    - pathlib.Path -> str
    - numpy scalars/arrays -> Python scalars / lists
    - torch tensors -> CPU lists
    - dict/list/tuple recursively

    Parameters
    ----------
    obj:
        Any Python object.

    Returns
    -------
    Any
        JSON-serializable equivalent (best effort).
    """
    # Path
    if isinstance(obj, Path):
        return str(obj)

    # Numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Torch
    if torch is not None:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()

    # Mappings
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Sequences (but not strings/bytes)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    # Basic JSON types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback: try json.dumps to see if it is already serializable
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)