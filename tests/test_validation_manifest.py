from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IDS_PATH = ROOT / "configs" / "validation" / "local_val_400_ids.txt"
MANIFEST_PATH = ROOT / "configs" / "validation" / "local_val_400_manifest.json"


def test_fixed_validation_manifest_matches_committed_ids() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    query_ids = IDS_PATH.read_text(encoding="utf-8").splitlines()

    assert len(query_ids) == 400
    assert len(set(query_ids)) == 400
    assert manifest["sample_count"] == 400
    assert manifest["source_record_count"] == 10834
    assert manifest["seed"] == 42
    assert manifest["first_id"] == query_ids[0]
    assert manifest["last_id"] == query_ids[-1]
    normalized_ids = ("\n".join(query_ids) + "\n").encode("utf-8")
    assert manifest["ids_sha256"] == hashlib.sha256(normalized_ids).hexdigest()
