from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_official_dataset import check_official_dataset
from scripts.visualize_repechage_samples import visualize_repechage_samples
from src.data.official import load_official_queries, resolve_official_root


def test_resolve_official_root_from_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIC_OFFICIAL_ROOT", str(tmp_path))
    assert resolve_official_root() == tmp_path.resolve()


def test_load_official_queries_rejects_missing_field(
    official_dataset_factory,
) -> None:
    root, records = official_dataset_factory(query_count=1, group_count=1)
    query_id = next(iter(records))
    del records[query_id]["depth"]
    (root / "queries" / "queries.json").write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_official_queries(root)


def test_check_official_dataset_passes_on_consistent_fixture(
    official_dataset_factory,
) -> None:
    root, _ = official_dataset_factory(query_count=3, group_count=2)

    report = check_official_dataset(
        root,
        expected_query_count=3,
        expected_image_group_count=2,
    )

    assert report["status"] == "PASS"
    assert report["summary"]["query_count"] == 3
    assert report["summary"]["unique_image_group_count"] == 2
    assert report["modalities"]["visible"]["shape_counts"] == {"12x16": 2}
    assert report["modalities"]["infrared"]["channel_counts"] == {"3": 2}
    assert report["modalities"]["depth"]["dtype_counts"] == {"uint16": 2}


def test_check_official_dataset_reports_missing_file(
    official_dataset_factory,
) -> None:
    root, records = official_dataset_factory(query_count=1, group_count=1)
    missing_path = root / next(iter(records.values()))["visible"]
    missing_path.unlink()

    report = check_official_dataset(
        root,
        expected_query_count=1,
        expected_image_group_count=1,
    )

    assert report["status"] == "FAIL"
    assert report["path_checks"]["visible"]["missing_reference_count"] == 1
    assert "referenced files are missing" in " ".join(report["errors"])


def test_check_official_dataset_reports_unreferenced_files(
    official_dataset_factory,
) -> None:
    root, _ = official_dataset_factory(
        query_count=1,
        group_count=1,
        extra_visible=1,
    )

    report = check_official_dataset(
        root,
        expected_query_count=1,
        expected_image_group_count=1,
    )

    assert report["status"] == "PASS"
    assert report["path_checks"]["visible"]["unreferenced_file_count"] == 1
    assert report["warnings"]


def test_check_official_dataset_writes_reports(
    official_dataset_factory,
    tmp_path: Path,
) -> None:
    root, _ = official_dataset_factory(query_count=1, group_count=1)
    json_report = tmp_path / "reports" / "quality.json"
    markdown_report = tmp_path / "reports" / "quality.md"

    report = check_official_dataset(
        root,
        expected_query_count=1,
        expected_image_group_count=1,
        json_report=json_report,
        markdown_report=markdown_report,
    )

    assert report["status"] == "PASS"
    assert json.loads(json_report.read_text(encoding="utf-8"))["status"] == "PASS"
    markdown = markdown_report.read_text(encoding="utf-8")
    assert "# Official Repechage Data Quality Report" in markdown
    assert "target example" not in markdown


def test_visualize_repechage_samples_generates_images_and_manifest(
    official_dataset_factory,
    tmp_path: Path,
) -> None:
    root, _ = official_dataset_factory(query_count=2, group_count=2)
    output_dir = tmp_path / "visualizations"

    report = visualize_repechage_samples(
        root,
        output_dir=output_dir,
        count=2,
        seed=42,
    )

    assert report["generated_count"] == 2
    assert len(list(output_dir.glob("*.jpg"))) == 2
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_count"] == 2
    assert len(manifest["samples"]) == 2
    assert "small_or_distant_target" in {
        category
        for sample in manifest["samples"]
        for category in sample["difficulty_categories"]
    }


def test_visualize_repechage_samples_accepts_specific_query_id(
    official_dataset_factory,
    tmp_path: Path,
) -> None:
    root, records = official_dataset_factory(query_count=2, group_count=2)
    query_id = sorted(records)[0]

    report = visualize_repechage_samples(
        root,
        output_dir=tmp_path / "visualizations",
        count=1,
        query_ids=[query_id],
    )

    assert report["generated_count"] == 1
    assert report["samples"][0]["query_id"] == query_id