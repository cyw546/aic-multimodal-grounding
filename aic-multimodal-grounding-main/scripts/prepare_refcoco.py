import argparse
import json
import os
from pathlib import Path
import random
from refer import REFER


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

        # 归一化到0~1 xyxy
        x1n = max(0.0, x1 / img_w)
        y1n = max(0.0, y1 / img_h)
        x2n = min(1.0, x2 / img_w)
        y2n = min(1.0, y2 / img_h)

        # 过滤非法框
        if not (0 <= x1n < x2n <= 1 and 0 <= y1n < y2n <= 1):
            continue

        split = ref["split"]
        if split not in ["train", "val"]:
            continue

        sample_id = f"{dataset_name}_{split}_{ref_id:08d}"
        query_text = ref["sentences"][0]["sent"].strip()
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
            "bbox": [round(x1n, 6), round(y1n, 6), round(x2n, 6), round(y2n, 6)],
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
    parser.add_argument("--data-root", type=str, help="refer数据集根目录")
    parser.add_argument("--image-root", type=str, help="coco2014图片文件夹")
    parser.add_argument("--out-dir", type=str, help="输出train.jsonl val.jsonl的目录")
    args = parser.parse_args()

    refer = REFER(data_root=args.data_root, dataset=args.dataset, splitBy="unc")
    convert_refcoco(refer, Path(args.image_root), Path(args.out_dir), dataset_name=args.dataset)


if __name__ == "__main__":
    main()