"""Convert RefCOCO JSONL into the project's unified JSONL format."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from src.metrics import validate_bbox

PROMPT_MARKER = 'describes: "'
PROMPT_SUFFIX = '".'

def extract_query(record: dict[str, Any], line_number: int) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"line {line_number}: conversations must be a list")
    prompts = [x.get("value") for x in conversations if isinstance(x, dict) and x.get("from") == "human"]
    if len(prompts) != 1 or not isinstance(prompts[0], str):
        raise ValueError(f"line {line_number}: expected exactly one human prompt")
    prompt = prompts[0].strip()
    start = prompt.find(PROMPT_MARKER)
    if start < 0 or not prompt.endswith(PROMPT_SUFFIX):
        raise ValueError(f"line {line_number}: unsupported prompt template")
    query = prompt[start + len(PROMPT_MARKER):-len(PROMPT_SUFFIX)].strip()
    if not query:
        raise ValueError(f"line {line_number}: empty referring expression")
    return query

def convert_record(record: dict[str, Any], line_number: int, source: str, split: str,
                   image_root: Path | None = None, require_images: bool = False) -> dict[str, Any]:
    image_name = record.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError(f"line {line_number}: image must be a non-empty string")
    image_path = Path(image_name.strip().replace("\\\\", "/"))
    if image_path.is_absolute() or ".." in image_path.parts:
        raise ValueError(f"line {line_number}: unsafe image path")
    if require_images:
        if image_root is None:
            raise ValueError("image_root is required when require_images=True")
        if not (image_root / image_path).is_file():
            raise FileNotFoundError(f"line {line_number}: missing image {image_path}")
    objects = record.get("objects")
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError(f"line {line_number}: expected exactly one object")
    bbox = validate_bbox(objects[0].get("bbox"), name=f"line {line_number} bbox")
    image_id, width, height = record.get("id"), record.get("width"), record.get("height")
    if not isinstance(image_id, int):
        raise ValueError(f"line {line_number}: id must be an integer")
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise ValueError(f"line {line_number}: invalid image width")
    if height is not None and (not isinstance(height, int) or height <= 0):
        raise ValueError(f"line {line_number}: invalid image height")
    sample_id = f"{source}_{split}_{line_number:08d}"
    return {"sample_id": sample_id, "source": source, "split": split, "query_id": sample_id,
            "query": extract_query(record, line_number), "visible_path": image_path.as_posix(),
            "infrared_path": None, "depth_path": None,
            "bbox": [round(float(x), 8) for x in bbox], "bbox_format": "xyxy_norm",
            "width": width, "height": height, "image_id": image_id}

def convert_jsonl(input_path: str | Path, output_path: str | Path, source: str = "refcoco",
                  split: str = "train", image_root: str | Path | None = None,
                  require_images: bool = False, limit: int | None = None) -> int:
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    root = Path(image_root).resolve() if image_root is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    try:
        with input_path.open(encoding="utf-8") as src, temporary.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"line {line_number}: record must be an object")
                item = convert_record(record, line_number, source, split, root, require_images)
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="refcoco"); parser.add_argument("--split", required=True)
    parser.add_argument("--image-root"); parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = convert_jsonl(args.input, args.output, args.source, args.split,
                          args.image_root, args.require_images, args.limit)
    print(f"converted {count} records -> {args.output}")

if __name__ == "__main__":
    main()
