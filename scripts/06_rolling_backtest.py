from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.rolling_train import run_rolling_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite-models", action="store_true", help="retrain rolling models even if artifacts already exist")
    parser.add_argument("--predict-workers", type=int, default=None, help="CPU worker count for prediction/evaluation")
    parser.add_argument("--windows", type=int, nargs="+", default=None, help="rolling windows to run, e.g. --windows 5 or --windows 5 8")
    args = parser.parse_args()

    config = load_config(args.config)
    run_rolling_backtest(config, overwrite_models=args.overwrite_models or None, predict_workers=args.predict_workers, windows=args.windows)
    print("rolling backtest completed")


if __name__ == "__main__":
    main()
