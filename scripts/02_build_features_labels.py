from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.feature_engineering import build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = build_features(config)
    if dataset.empty:
        print("feature dataset rows: saved as daily feature parts")
    else:
        print(f"feature dataset rows: {len(dataset):,}")


if __name__ == "__main__":
    main()
