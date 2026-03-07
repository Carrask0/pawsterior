
import torch
# torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision("high")
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import json


def train_model(
    model,
    model_type,  # "Velocity" or "X0X1"
    theta_train,
    x_train,
    theta_val,
    x_val,
    config,
    task_name=None,
    n_samples=None,
    model_key=None,     # "velocity" or "x0x1"
    init_dist="gaussian",
    grid_index=None,
):
    """
    Trains a Flow Matching / X0X1 model.

    Args:
        model: wrapper with .prediction_net and .compute_loss(...)
        model_type: "Velocity" or "X0X1"
        theta_train, x_train, theta_val, x_val: arrays/tensors
        config: dict
        learning_rate, batch_size: optional overrides for the training config
        ckpt_suffix: appended to checkpoint filename

    Returns:
        history dict with 'train_loss', 'val_loss', 'best_val', 'best_epoch', 'ckpt_path'
    """

    # -----------------------------
    # Prep: device, tensors, params
    # -----------------------------
    device = model.device

    def to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr
        arr = np.asarray(arr)
        return torch.from_numpy(arr).float()

    x_train = to_tensor(x_train).contiguous()
    x_val = to_tensor(x_val).contiguous()
    theta_train = to_tensor(theta_train).contiguous()
    theta_val = to_tensor(theta_val).contiguous()

    if device.type == "mps":
        theta_train = theta_train.contiguous().to(device)
        x_train     = x_train.contiguous().to(device)
        theta_val   = theta_val.contiguous().to(device)
        x_val       = x_val.contiguous().to(device)

    train_ds = TensorDataset(theta_train, x_train)
    val_ds = TensorDataset(theta_val, x_val)

    model.prediction_net.train()

    # Set training params
    lr = float(config["params"]["learning_rate"])
    hidden_dim = int(model.prediction_net.hidden_dim)
    num_blocks = int(model.prediction_net.num_blocks)
    num_epochs = int(config["train"]["num_epochs"])
    bs = int(config["train"]["batch_size"])
    patience = int(config["train"]["early_stopping_patience"])  # None disables
    min_delta = float(config["train"]["early_stopping_min_delta"])
    optimizer = torch.optim.Adam(model.prediction_net.parameters(), lr=lr)
    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    # Define data loaders 
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=0,          # no multiproc, no worker startup cost
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # Training logging
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improve = 0

    # Output checkpoint path
    from ..paths import ckpt_path as _ckpt_path, ckpt_meta_path as _meta_path

    if task_name is None:
        raise ValueError("train_model requires task_name (storage key).")
    if n_samples is None:
        raise ValueError("train_model requires n_samples.")
    if model_key is None:
        raise ValueError("train_model requires model_key in {'x0x1','velocity'}.")

    ckpt_path = _ckpt_path(model_key, init_dist, task_name, int(n_samples), grid_index)
    meta_path = _meta_path(model_key, init_dist, task_name, int(n_samples), grid_index)

    os.makedirs(ckpt_path.parent, exist_ok=True)


    train_size = theta_train.shape[0]
    val_size = theta_val.shape[0]

    print(f"\n{'='*60}")
    print(f"Training {model_type} Model")
    print(
        f"device={device} | N_train={train_size} | N_val={val_size} | bs={bs} | lr={lr:.2e}"
    )
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        # -----------------------------
        # Train epoch
        # -----------------------------
        model.prediction_net.train()
        train_loss_sum = 0.0

        for theta_b, x_b in train_loader:
            
            # DEBUG
            def _finite(name, t):
                ok = torch.isfinite(t).all().item()
                if not ok:
                    bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:10]
                    print(f"[BAD] {name} finite={ok} dtype={t.dtype} device={t.device} shape={tuple(t.shape)}")
                    print("[BAD] first idx:", bad.tolist())
                    raise ValueError(f"{name} contains NaN/Inf")

            # 1) check on CPU BEFORE any .to(device)
            _finite("theta_b (cpu)", theta_b)
            _finite("x_b (cpu)", x_b)


            if device.type != "mps":
                theta_b = theta_b.contiguous().to(device, non_blocking=True)
                x_b     = x_b.contiguous().to(device, non_blocking=True)

            # 2) check AFTER transfer
            _finite("theta_b (device)", theta_b)
            _finite("x_b (device)", x_b)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                loss, _ = model.compute_loss(theta_b, x_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prediction_net.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(train_loader))

        model.prediction_net.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for theta_b, x_b in val_loader:
                theta_b = theta_b.to(device, non_blocking=True)
                x_b     = x_b.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                    vloss, _ = model.compute_loss(theta_b, x_b)
                val_loss_sum += float(vloss.item())
        val_loss = val_loss_sum / max(1, len(val_loader))

        # LR scheduler on validation loss
        scheduler.step(val_loss)

        # Logging
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Checkpoint if improved
        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            state = {
                "model_type": model_type,
                "state_dict": model.prediction_net.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
            }
            torch.save(state, str(ckpt_path))

            meta = {
                "task": task_name,
                "n_samples": int(n_samples),
                "grid_index": grid_index,
                "model_key": model_key,
                "init_dist": init_dist,
                "model_type_str": model_type,
                "hparams": {
                    "hidden_dim": hidden_dim,
                    "num_blocks": num_blocks,
                    "learning_rate": lr,
                    "batch_size": bs,
                },
                "best_epoch": int(epoch),
                "best_val": float(val_loss),
            }
            with open(meta_path, "w") as fp:
                json.dump(meta, fp, indent=2)
        else:
            epochs_without_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 50 == 0 or epoch == num_epochs - 1: 
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | lr {lr_now:.2e} | "
                f"best {best_val:.4f} @ {best_epoch+1 if best_epoch>=0 else 0}"
            )

        # Optional early stopping
        if (patience is not None) and (epochs_without_improve >= int(patience)):
            print(
                f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)."
            )
            break

    history.update(
        {"best_val": best_val, "best_epoch": best_epoch, "ckpt_path": ckpt_path}
    )
    return history
