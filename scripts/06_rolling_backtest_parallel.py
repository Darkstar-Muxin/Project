from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def _visible_devices(train_workers: int, devices_arg: str | None) -> list[str]:
    if devices_arg:
        devices = [part.strip() for part in devices_arg.split(",") if part.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        devices = [part.strip() for part in visible.split(",") if part.strip()]
    if not devices:
        devices = [str(i) for i in range(max(train_workers, 1))]
    return devices


def _train_worker(payload: dict[str, Any]) -> dict[str, Any]:
    device = str(payload["device"])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    config = payload["config"]
    config["ive_device"] = "cuda"
    from src.rolling_train import train_rolling_task

    print(
        f"[parallel-train] device={device} split={payload['task']['split']} "
        f"test_date={payload['task']['test_date']} window={payload['task']['window']}",
        flush=True,
    )
    return train_rolling_task(
        config,
        payload["task"],
        payload["dataset_path"],
        payload["all_columns"],
        overwrite_models=payload["overwrite_models"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-workers", type=int, default=None, help="number of parallel training workers")
    parser.add_argument("--predict-workers", type=int, default=None, help="CPU worker count for prediction/evaluation")
    parser.add_argument("--devices", default=None, help="comma-separated GPU ids; defaults to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--overwrite-models", action="store_true", help="retrain rolling models even if artifacts already exist")
    parser.add_argument("--windows", type=int, nargs="+", default=None, help="rolling windows to run, e.g. --windows 5 or --windows 5 8")
    parser.add_argument("--months", nargs="+", default=None, help="target months to run, e.g. --months 202604")
    parser.add_argument("--train-only", action="store_true", help="train rolling models and skip prediction/evaluation")
    parser.add_argument("--predict-only", action="store_true", help="skip training and run prediction/evaluation from existing models")
    parser.add_argument("--skip-existing-predictions", action="store_true", help="skip prediction parts whose outputs already exist")
    parser.add_argument("--skip-final-reports", action="store_true", help="write prediction parts only; skip final report aggregation")
    parser.add_argument("--aggregate-only", action="store_true", help="skip train/predict and aggregate existing prediction parts into final reports")
    parser.add_argument(
        "--predict-unit",
        choices=["day", "group"],
        default=None,
        help="parallel prediction unit; group balances large liquidity groups but reads each day more than once",
    )
    args = parser.parse_args()
    if args.train_only and args.predict_only:
        parser.error("--train-only and --predict-only cannot be used together")

    config = load_config(args.config)
    from src.rolling_train import _schema_columns, aggregate_rolling_prediction_parts, build_rolling_tasks, run_rolling_predictions

    if args.aggregate_only:
        aggregate_rolling_prediction_parts(config, windows=args.windows, months=args.months)
        print("parallel rolling aggregation completed")
        return

    tasks, dataset_path, _ = build_rolling_tasks(config, windows=args.windows, months=args.months)
    all_columns = _schema_columns(dataset_path)
    train_workers = int(args.train_workers or config.get("rolling_train_workers", 1))
    train_workers = max(train_workers, 1)
    devices = _visible_devices(train_workers, args.devices)
    if args.predict_only:
        print(f"[parallel-train] predict-only; skip training tasks={len(tasks)}", flush=True)
        run_rolling_predictions(
            config,
            tasks,
            dataset_path,
            all_columns,
            predict_workers=args.predict_workers,
            predict_unit=args.predict_unit,
            skip_existing_predictions=args.skip_existing_predictions,
            write_final_reports=not args.skip_final_reports,
        )
        print("parallel rolling prediction completed")
        return
    payloads = [
        {
            "config": dict(config),
            "task": task,
            "dataset_path": str(dataset_path),
            "all_columns": all_columns,
            "overwrite_models": args.overwrite_models or bool(config.get("rolling_overwrite_models", False)),
            "device": devices[i % len(devices)],
        }
        for i, task in enumerate(tasks)
    ]
    print(f"[parallel-train] tasks={len(payloads)} workers={train_workers} devices={devices}", flush=True)
    if train_workers == 1:
        for payload in payloads:
            _train_worker(payload)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=train_workers, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_train_worker, payloads):
                pass

    if args.train_only:
        print("[parallel-train] train-only completed; skip prediction/evaluation", flush=True)
        return
    print(f"[parallel-train] training completed; start prediction/evaluation", flush=True)
    run_rolling_predictions(
        config,
        tasks,
        dataset_path,
        all_columns,
        predict_workers=args.predict_workers,
        predict_unit=args.predict_unit,
        skip_existing_predictions=args.skip_existing_predictions,
        write_final_reports=not args.skip_final_reports,
    )
    print("parallel rolling backtest completed")


if __name__ == "__main__":
    main()
