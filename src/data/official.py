from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_QUERIES_RELATIVE_PATH = Path("queries") / "queries.json"
REQUIRED_QUERY_FIELDS = ("visible", "infrared", "depth", "query")


def resolve_official_root(
    data_root: str | Path | None = None,
    *,
    env_var: str = "AIC_OFFICIAL_ROOT",
) -> Path:
    """Resolve an official dataset root from an argument or environment variable."""

    value = data_root if data_root is not None else os.environ.get(env_var)
    if value is None or not str(value).strip():
        raise ValueError(
            f"Official data root is not set. Pass data_root or set {env_var}."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Official data root does not exist: {root}")
    return root


def resolve_queries_path(
    data_root: str | Path,
    queries_path: str | Path | None = None,
) -> Path:
    """Resolve the queries JSON path, supporting an optional CLI override."""

    root = Path(data_root).expanduser().resolve()
    if queries_path is None:
        return root / DEFAULT_QUERIES_RELATIVE_PATH
    path = Path(queries_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_relative_path(
    data_root: str | Path,
    value: Any,
    *,
    field: str,
    query_id: str,
) -> Path:
    """Resolve a safe, repository-relative image path from an official record."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{query_id}: {field} must be a non-empty string")
    root = Path(data_root).expanduser().resolve()
    relative = Path(value.strip().replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{query_id}: unsafe {field} path: {value!r}")
    return root / relative


def load_official_queries(
    data_root: str | Path | None = None,
    queries_path: str | Path | None = None,
    *,
    env_var: str = "AIC_OFFICIAL_ROOT",
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    """Load and validate the common official JSON schema.

    The reader intentionally works for sample, preliminary, and repechage data.
    It validates fields that are shared by all splits and leaves bbox handling
    to :class:`MultimodalGroundingDataset`.
    """

    root = resolve_official_root(data_root, env_var=env_var)
    json_path = resolve_queries_path(root, queries_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Official queries JSON does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Official queries JSON must be an object keyed by Query ID")

    records: dict[str, dict[str, Any]] = {}
    for query_id, record in payload.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Every official record must have a non-empty string Query ID")
        if not isinstance(record, dict):
            raise ValueError(f"{query_id}: record must be an object")
        missing = [field for field in REQUIRED_QUERY_FIELDS if field not in record]
        if missing:
            raise ValueError(f"{query_id}: missing required fields {missing}")
        if not isinstance(record["query"], str) or not record["query"].strip():
            raise ValueError(f"{query_id}: query must be a non-empty string")
        records[query_id] = record
    return root, json_path, records