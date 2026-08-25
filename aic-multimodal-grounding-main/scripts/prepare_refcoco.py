import argparse
import json
import os
from pathlib import Path
import random
from refer import REFER


def is_bbox_valid(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False, None
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False, None
    out = [
        x1 / img_w,
        y1 / img_h,
        x2 / img_w,
        y2 / img_h
    ]
    for v in out:
        if v < 0 or v > 1:
            return False, None
    return True, out


def convert_refcoco(refer: REFER, image_root: Path, out_dir: Path, dataset_name="refcoco"):
    out_dir.mkdir(exist_ok=True, parents=True)
    splits = {"train": [], "val": []}

    for ref_id in refer.getRefIds():
        ref = refer.Refs[ref_id]
        ann = refer.Anns[ref["ann_id"]]
        img = refer.Imgs[ref["image_id"]]

        # 原始coco bbox [x,y,w,h] 像素
        x, y, w, h = ann["bbox"]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        img_w = img["width"]
        img_h = img["height"]

        ok, norm_bbox = is_bbox_valid([x1, y1, x2, y2], img_w, img_h)
        if not ok:
            continue

        split = ref["split"]
        if split not in ["train", "val"]:
            continue

        sample_id = f"{dataset_name}_{split}_{ref_id:08d}"
        query_text = ref["sentences"][0]["sent"].strip()
        # 过滤空query
        if not query_text:
            continue

        visible_path = os.path.basename(img["file_name"])

        item = {
            "sample_id": sample_id,
            "source": dataset_name,
            "split": split,
            "query_id": str(ref_id),
            "query": query_text,
            "visible_path": visible_path,
            "infrared_path": None,
            "depth_path": None,
            "bbox": [round(norm_bbox[0], 6), round(norm_bbox[1], 6), round(norm_bbox[2], 6), round(norm_bbox[3], 6)],
            "bbox_format": "xyxy_norm",
            "width": img_w,
            "height": img_h
        }
        splits[split].append(item)

    # 写jsonl
    for sp, samples in splits.items():
        out_file = out_dir / f"{sp}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"完成！输出目录：{out_dir}")
    print(f"train:{len(splits['train'])}  val:{len(splits['val'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="refcoco", choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/aic_grounding/data/public/refcoco", help="refer数据集根目录")
    parser.add_argument("--image-root", type=str, help="coco2014图片文件夹")
    parser.add_argument("--out-dir", type=str, help="输出train.jsonl val.jsonl的目录")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"【错误】标注文件夹不存在：{data_root}")
        print("请放入RefCOCO原始标注文件到此路径")
        raise SystemExit(1)

    refer = REFER(data_root=str(data_root), dataset=args.dataset, splitBy="unc")
    convert_refcoco(refer, Path(args.image_root), Path(args.out_dir), dataset_name=args.dataset)


if __name__ == "__main__":
    main()
