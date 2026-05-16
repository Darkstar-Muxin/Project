from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config
from src.data_loader import load_raw_parquet_files
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/outputs/schema_summary.json")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    summary = load_raw_parquet_files(args.data_dir, config)
    write_json(args.out, summary)
    print(f"schema summary written to {args.out}")


if __name__ == "__main__":
    main()
