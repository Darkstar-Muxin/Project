from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.utils import resolve_path


FIELD_GROUPS = {
    "stock_code": "stock_code_candidates",
    "datetime": "datetime_candidates",
    "date": "date_candidates",
    "price": "price_candidates",
    "volume": "volume_candidates",
    "amount": "amount_candidates",
}


def _is_excluded(path: Path, exclude_dirs: list[str] | None) -> bool:
    if not exclude_dirs:
        return False
    resolved = path.resolve()
    for item in exclude_dirs:
        ex = resolve_path(item).resolve()
        if resolved == ex or ex in resolved.parents:
            return True
    return False


def find_parquet_files(data_dir: str | Path, exclude_dirs: list[str] | None = None) -> list[Path]:
    root = resolve_path(data_dir)
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*.parquet"):
        if not _is_excluded(path, exclude_dirs):
            files.append(path)
    return sorted(files)


def read_parquet_sample(path: str | Path, nrows: int = 5) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(resolve_path(path))
    batch = next(parquet_file.iter_batches(batch_size=nrows))
    return batch.to_pandas()


def infer_column_mapping(columns: list[str], config: dict[str, Any]) -> dict[str, str | None]:
    lower_to_original = {str(col).lower(): str(col) for col in columns}
    mapping: dict[str, str | None] = {}
    for canonical, config_key in FIELD_GROUPS.items():
        found = None
        for candidate in config.get(config_key, []):
            found = lower_to_original.get(str(candidate).lower())
            if found is not None:
                break
        mapping[canonical] = found
    return mapping


def summarize_parquet_file(path: str | Path, config: dict[str, Any], sample_rows: int = 5) -> dict[str, Any]:
    p = resolve_path(path)
    try:
        parquet_file = pq.ParquetFile(p)
        columns = parquet_file.schema_arrow.names
        sample = read_parquet_sample(p, sample_rows)
        return {
            "path": str(p),
            "rows": parquet_file.metadata.num_rows,
            "columns": columns,
            "column_mapping": infer_column_mapping(columns, config),
            "sample": sample.to_dict(orient="records"),
            "error": None,
        }
    except Exception as exc:
        return {"path": str(p), "rows": None, "columns": [], "column_mapping": {}, "sample": [], "error": str(exc)}


def load_raw_parquet_files(data_dir: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    files = find_parquet_files(data_dir, config.get("exclude_dirs", []))
    return [summarize_parquet_file(path, config) for path in files]
