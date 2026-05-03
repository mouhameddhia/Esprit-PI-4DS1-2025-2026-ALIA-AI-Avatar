from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SequenceDataset
from src.metrics import classification_metrics
from src.model import StressLSTM
from src.preprocess import (
    DataConfig,
    PreprocessConfig,
    apply_feature_scaler,
    build_sequences,
    fit_feature_scaler,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def group_or_random_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
    group_split: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(X))
    if group_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss.split(idx, y, groups=groups))

        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=random_state + 1,
        )
        train_idx_rel, val_idx_rel = next(
            gss2.split(train_val_idx, y[train_val_idx], groups=groups[train_val_idx])
        )
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
    else:
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=y[train_val_idx],
            random_state=random_state,
        )
    return train_idx, val_idx, test_idx


def build_loss(y_train: np.ndarray, class_weighting: bool, device: torch.device) -> nn.Module:
    if not class_weighting:
        return nn.CrossEntropyLoss()
    counts = np.bincount(y_train)
    if len(counts) < 2:
        return nn.CrossEntropyLoss()
    weights = counts.sum() / np.maximum(counts, 1)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    all_preds, all_targets = [], []
    total_loss = 0.0

    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    data_cfg = DataConfig(
        subject_col=cfg["data"]["subject_col"],
        label_col=cfg["data"]["label_col"],
        frame_col=cfg["data"]["frame_col"],
        feature_cols=cfg["data"]["feature_cols"],
        sequence_length=cfg["data"]["sequence_length"],
        stride=cfg["data"]["stride"],
    )
    prep_cfg = PreprocessConfig(**cfg["preprocessing"])

    df = pd.read_csv(cfg["data"]["input_csv"])
    X, y, groups = build_sequences(df, data_cfg, prep_cfg)
    if len(X) == 0:
        raise RuntimeError("No sequences were generated. Check sequence_length and input data.")

    train_idx, val_idx, test_idx = group_or_random_split(
        X,
        y,
        groups,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
        group_split=cfg["data"]["group_split"],
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if prep_cfg.normalize == "zscore":
        scaler = fit_feature_scaler(X_train)
        X_train = apply_feature_scaler(X_train, scaler)
        X_val = apply_feature_scaler(X_val, scaler)
        X_test = apply_feature_scaler(X_test, scaler)
    else:
        scaler = None

    model_cfg = dict(cfg["model"])
    model_cfg["input_dim"] = int(X_train.shape[-1])

    device = choose_device(cfg["training"]["device"])
    model = StressLSTM(**model_cfg).to(device)

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    criterion = build_loss(y_train, cfg["training"]["class_weighting"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["training"]["scheduler_factor"],
        patience=cfg["training"]["scheduler_patience"],
    )

    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_state = None
    best_f1 = -1.0
    wait = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=cfg["training"]["grad_clip"],
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)

        scheduler.step(val_metrics["f1"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, criterion, device)

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    if scaler is not None:
        np.save(output_dir / "scaler_mean.npy", scaler.mean_)
        np.save(output_dir / "scaler_scale.npy", scaler.scale_)

    summary = {
        "best_val_f1": best_f1,
        "test_metrics": test_metrics,
        "model": model_cfg,
        "data": {
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "input_csv": cfg["data"]["input_csv"],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved model and metrics to", output_dir)


if __name__ == "__main__":
    main()
