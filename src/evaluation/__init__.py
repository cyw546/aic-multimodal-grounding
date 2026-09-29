"""Unified model evaluation and candidate diagnosis utilities."""

from .evaluator import (
    DEFAULT_COVERAGE_KS,
    evaluate_prediction_records,
    freeze_query_ids,
    load_candidate_records,
    load_prediction_records,
    load_query_ids,
    load_validation_records,
    recommend_priority,
    select_validation_records,
    sha256_file,
    summarize_rows,
    visualize_failures,
)

__all__ = [
    "DEFAULT_COVERAGE_KS",
    "evaluate_prediction_records",
    "freeze_query_ids",
    "load_candidate_records",
    "load_prediction_records",
    "load_query_ids",
    "load_validation_records",
    "recommend_priority",
    "select_validation_records",
    "sha256_file",
    "summarize_rows",
    "visualize_failures",
]
