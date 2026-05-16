from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.preprocess import preprocess_raw_data
from src.stock_classification import classify_stocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    minute_df = preprocess_raw_data(config)
    group_df = classify_stocks(minute_df, config)
    print(f"minute rows: {len(minute_df):,}")
    print(f"stock groups: {len(group_df):,}")


if __name__ == "__main__":
    main()
