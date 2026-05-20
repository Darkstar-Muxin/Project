from __future__ import annotations

import argparse
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.config import load_config
from src.feature_engineering import _build_one_day_features, _minute_part_paths
from src.utils import resolve_path


def _build_one(args: tuple[str, list[str], str, dict[str, Any], bool]) -> tuple[str, int, str]:
    current_path_str, history_path_strs, out_path_str, config, overwrite = args
    current_path = Path(current_path_str)
    history_paths = [Path(item) for item in history_path_strs]
    out_path = Path(out_path_str)
    if out_path.exists() and not overwrite:
        return current_path.name, -1, "cached"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_df = _build_one_day_features(current_path, history_paths, config)
    day_df.to_parquet(out_path, index=False)
    return current_path.name, len(day_df), "built"


def _make_tasks(
    config: dict[str, Any],
    overwrite: bool,
    months: set[str] | None = None,
) -> list[tuple[str, list[str], str, dict[str, Any], bool]]:
    part_paths = _minute_part_paths(config)
    if not part_paths:
        raise FileNotFoundError("No minute part files found. Run scripts/01_preprocess.py first.")

    feature_parts_dir = resolve_path(config.get("feature_parts_dir", "data/features/model_parts"))
    max_history_days = int(config.get("feature_history_days", 20))
    tasks: list[tuple[str, list[str], str, dict[str, Any], bool]] = []
    for idx, current_path in enumerate(part_paths):
        if months is not None and current_path.stem[:6] not in months:
            continue
        history_paths = part_paths[max(0, idx - max_history_days) : idx]
        out_path = feature_parts_dir / current_path.name
        tasks.append(
            (
                str(current_path),
                [str(path) for path in history_paths],
                str(out_path),
                config,
                overwrite,
            )
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(2, (os.cpu_count() or 2) // 2)),
        help="Number of day-level worker processes. Keep this small because each worker reads history day files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data/features/model_parts/YYYYMMDD.parquet files.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        help="Optional YYYYMM filters, for example: --months 202602 202603. History days before those months are still used when available.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    months = {str(month) for month in args.months} if args.months else None
    tasks = _make_tasks(config, overwrite=bool(args.overwrite), months=months)
    workers = max(1, int(args.workers))
    month_text = "all configured months" if months is None else ",".join(sorted(months))
    print(
        f"building {len(tasks)} feature parts for months={month_text} "
        f"with workers={workers}, overwrite={bool(args.overwrite)}",
        flush=True,
    )

    built = 0
    cached = 0
    total_rows = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for name, rows, status in tqdm(
            pool.imap_unordered(_build_one, tasks),
            total=len(tasks),
            desc="build feature parts multiprocessing",
        ):
            if status == "cached":
                cached += 1
            else:
                built += 1
                total_rows += rows
            print(f"[feature-parallel] {status} {name} rows={rows if rows >= 0 else '-'}", flush=True)

    print(f"feature parts completed: built={built}, cached={cached}, rows={total_rows:,}", flush=True)


if __name__ == "__main__":
    main()
