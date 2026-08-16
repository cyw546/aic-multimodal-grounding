import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.prepare_refcoco import convert_jsonl
from src.data import PublicGroundingDataset

def raw(query="person on the right", bbox=None, image="COCO_train2014_000000000001.jpg"):
    return {"id": 1, "image": image, "width": 8, "height": 6,
            "conversations": [{"from": "human", "value":
                f'Please carefully check the image and detect the object this sentence describes: "{query}".'}],
            "objects": [{"bbox": bbox or [0.1, 0.2, 0.8, 0.9]}]}

def write(path: Path, records):
    path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")

def test_preserves_every_expression(tmp_path):
    source, output = tmp_path / "raw.json", tmp_path / "train.jsonl"
    write(source, [raw("first person"), raw("second person")])
    assert convert_jsonl(source, output, split="train") == 2
    records = [json.loads(x) for x in output.read_text().splitlines()]
    assert [x["query"] for x in records] == ["first person", "second person"]
    assert records[0]["sample_id"] != records[1]["sample_id"]

def test_rejects_invalid_bbox(tmp_path):
    source = tmp_path / "raw.json"
    write(source, [raw(bbox=[0.8, 0.2, 0.1, 0.9])])
    with pytest.raises(ValueError):
        convert_jsonl(source, tmp_path / "out.jsonl", split="train")

def test_requires_images_when_requested(tmp_path):
    source = tmp_path / "raw.json"
    write(source, [raw()])
    with pytest.raises(FileNotFoundError):
        convert_jsonl(source, tmp_path / "out.jsonl", split="val",
                      image_root=tmp_path / "images", require_images=True)

def test_dataset_reads_rgb(tmp_path):
    root = tmp_path / "images"; root.mkdir()
    name = "COCO_train2014_000000000001.jpg"
    image = np.zeros((6, 8, 3), dtype=np.uint8); image[:, :, 2] = 255
    assert cv2.imwrite(str(root / name), image)
    source, output = tmp_path / "raw.json", tmp_path / "val.jsonl"
    write(source, [raw(image=name)])
    convert_jsonl(source, output, split="val", image_root=root, require_images=True)
    sample = PublicGroundingDataset(output, root)[0]
    assert sample["visible"].shape == (6, 8, 3)
    assert sample["visible"][0, 0].tolist() == [254, 0, 0]
    assert sample["infrared"] is None and sample["depth"] is None
    assert np.allclose(sample["bbox"], [0.1, 0.2, 0.8, 0.9])

def test_accepts_missing_dimensions(tmp_path):
    record = raw(); record.pop("width"); record.pop("height")
    source, output = tmp_path / "raw.json", tmp_path / "val.jsonl"
    write(source, [record])
    assert convert_jsonl(source, output, split="val") == 1
    converted = json.loads(output.read_text())
    assert converted["width"] is None and converted["height"] is None
