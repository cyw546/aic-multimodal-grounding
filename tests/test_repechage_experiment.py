import pytest

from scripts.run_repechage_rgb_experiment import (
    select_visualization_ids,
    summarize_details,
)


def detail(query_id, stage, score, bbox, runtime=1.0):
    return {
        "query_id": query_id,
        "stage": stage,
        "score": score,
        "bbox": bbox,
        "runtime_seconds": runtime,
    }


def test_summarize_details_counts_each_result_stage():
    summary = summarize_details([
        detail("a", "primary", 0.8, [0.1, 0.1, 0.5, 0.5], 1.0),
        detail("b", "fallback_threshold", 0.4, [0.2, 0.2, 0.6, 0.6], 2.0),
        detail("c", "default_box", 0.0, [0.25, 0.25, 0.75, 0.75], 3.0),
    ])
    assert summary["sample_count"] == 3
    assert summary["primary_success_count"] == 1
    assert summary["primary_empty_count"] == 2
    assert summary["fallback_success_count"] == 1
    assert summary["default_box_count"] == 1
    assert summary["mean_score"] == pytest.approx(0.4)
    assert summary["total_runtime_seconds"] == pytest.approx(6.0)


def test_visualization_selection_prioritizes_default_and_fallback():
    query_ids = ["normal", "fallback", "default"]
    configs = {
        "one": [
            detail("normal", "primary", 0.8, [0.1, 0.1, 0.5, 0.5]),
            detail("fallback", "fallback_threshold", 0.4, [0.2, 0.2, 0.6, 0.6]),
            detail("default", "default_box", 0.0, [0.25, 0.25, 0.75, 0.75]),
        ],
        "two": [
            detail("normal", "primary", 0.7, [0.1, 0.1, 0.5, 0.5]),
            detail("fallback", "primary", 0.5, [0.2, 0.2, 0.6, 0.6]),
            detail("default", "primary", 0.2, [0.3, 0.3, 0.7, 0.7]),
        ],
    }
    assert select_visualization_ids(query_ids, configs, 2) == [
        "default",
        "fallback",
    ]
