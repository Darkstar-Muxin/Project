from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for training. Install with: pip install torch") from exc

from src.config import ensure_dirs
from src.ive_dataset import (
    IVEDataset,
    Normalizer,
    build_vocab,
    fit_normalizer,
    get_ive_feature_columns,
    metadata_from_training,
)
from src.ive_model import IVEModel
from src.utils import resolve_path, write_json


def filter_by_months(df: pd.DataFrame, months: list[int | str] | None) -> pd.DataFrame:
    if not months:
        return df
    month_set = {str(month) for month in months}
    dt = pd.to_datetime(df["datetime"])
    return df.loc[dt.dt.strftime("%Y%m").isin(month_set)].copy()


def vwap_base(df: pd.DataFrame) -> pd.Series:
    if "vwap" in df.columns:
        base = pd.to_numeric(df["vwap"], errors="coerce")
    elif "close" in df.columns:
        base = pd.to_numeric(df["close"], errors="coerce")
    else:
        base = pd.Series(np.nan, index=df.index)
    return base.replace(0, np.nan)


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("ive_device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _train_loss(outputs: dict[str, torch.Tensor], volume_target: torch.Tensor, vwap_target: torch.Tensor) -> torch.Tensor:
    mu = outputs["volume_mu"]
    sigma = torch.nn.functional.softplus(outputs["volume_log_sigma"]) + 1e-4
    volume_nll = 0.5 * (((volume_target - mu) / sigma) ** 2 + 2 * torch.log(sigma))
    vwap_loss = torch.nn.functional.smooth_l1_loss(outputs["vwap_return"], vwap_target, reduction="none")
    return volume_nll.mean() + vwap_loss.mean()


def _fit_group_model(
    group_df: pd.DataFrame,
    group_path: Path,
    feature_columns: list[str],
    horizons: list[int],
    stock_vocab: dict[str, int],
    group_vocab: dict[str, int],
    normalizer: Normalizer,
    config: dict[str, Any],
) -> None:
    context_length = int(config.get("context_length", 390))
    dataset = IVEDataset(
        group_df,
        feature_columns,
        horizons,
        stock_vocab,
        group_vocab,
        normalizer,
        context_length=context_length,
    )
    if len(dataset) == 0:
        print(f"[train] skip {group_path.name}: no valid rows", flush=True)
        return
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("ive_batch_size", 256)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    device = _device(config)
    model = IVEModel(
        num_features=len(feature_columns),
        num_stocks=max(len(stock_vocab), 1),
        num_groups=max(len(group_vocab), 1),
        num_horizons=len(horizons),
        d_model=int(config.get("ive_d_model", 64)),
        nhead=int(config.get("ive_nhead", 4)),
        num_layers=int(config.get("ive_num_layers", 2)),
        dropout=float(config.get("ive_dropout", 0.1)),
        max_len=context_length,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("ive_learning_rate", 3e-4)),
        weight_decay=float(config.get("ive_weight_decay", 1e-4)),
    )
    epochs = int(config.get("ive_epochs", 5))
    print(f"[train] group={group_path.name} rows={len(dataset):,} device={device} epochs={epochs}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["x"].to(device),
                batch["stock_id"].to(device),
                batch["group_id"].to(device),
                batch["padding_mask"].to(device),
            )
            loss = _train_loss(outputs, batch["volume_target"].to(device), batch["vwap_target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[train] group={group_path.name} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)

    group_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), group_path / "ive_model.pt")
    meta = metadata_from_training(group_df, feature_columns, horizons, stock_vocab, group_vocab, normalizer, context_length)
    meta.update(
        {
            "d_model": int(config.get("ive_d_model", 64)),
            "nhead": int(config.get("ive_nhead", 4)),
            "num_layers": int(config.get("ive_num_layers", 2)),
            "dropout": float(config.get("ive_dropout", 0.1)),
            "model_type": "ive_transformer",
        }
    )
    write_json(group_path / "model_meta.json", meta)
    joblib.dump(feature_columns, group_path / "feature_columns.joblib")
    joblib.dump(stock_vocab, group_path / "stock_vocab.joblib")
    joblib.dump(group_vocab, group_path / "group_vocab.joblib")
    joblib.dump(normalizer, group_path / "normalizer.joblib")


def train_ive_models(
    train_df: pd.DataFrame,
    model_root: str | Path,
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> list[str]:
    horizons = [int(h) for h in config["horizons"]]
    train_df = train_df.sort_values(["stock_code", "datetime"]).reset_index(drop=True).copy()
    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    if feature_columns is None:
        feature_columns = get_ive_feature_columns(train_df, horizons)
    stock_vocab = build_vocab(train_df["stock_code"])
    group_vocab = build_vocab(train_df["liquidity_group"])
    normalizer = fit_normalizer(train_df, feature_columns)
    model_root = resolve_path(model_root)
    trained_groups: list[str] = []
    for group_name, group_df in train_df.groupby("liquidity_group", sort=True):
        if pd.isna(group_name):
            continue
        group_path = model_root / str(group_name)
        _fit_group_model(group_df, group_path, feature_columns, horizons, stock_vocab, group_vocab, normalizer, config)
        trained_groups.append(str(group_name))
    return trained_groups


def train_models(config: dict[str, Any]) -> None:
    ensure_dirs(config)
    dataset_path = resolve_path(config["feature_data_dir"]) / "model_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    df = filter_by_months(df, config.get("train_months"))
    if df.empty:
        raise ValueError("No training rows after applying train_months filter")
    train_ive_models(df, resolve_path(config["model_dir"]), config)
