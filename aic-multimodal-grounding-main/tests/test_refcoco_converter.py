import json
from pathlib import Path
import random


def check_jsonl(path: Path):
    samples = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            samples.append(s)

    ok = True
    for idx, item in enumerate(samples):
        bbox = item["bbox"]
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            print(f"非法bbox，行{idx}: {bbox}")
            ok = False
        if not isinstance(item["query"], str) or len(item["query"]) == 0:
            print(f"query为空，行{idx}")
            ok = False
    print(f"文件 {path} 总样本数 {len(samples)}")

    # 随机抽取20条打印检查
    print("\n====随机抽查20条样本====")
    pick = random.sample(samples, k=20)
    for p in pick:
        print(f"sample_id:{p['sample_id']}, query:{p['query']}, bbox:{p['bbox']}")

    if ok:
        print("\n✅全部校验通过")
    else:
        raise AssertionError("存在非法样本")
    return samples


if __name__ == "__main__":
    check_jsonl(Path("./output_refcoco/train.jsonl"))
    check_jsonl(Path("./output_refcoco/val.jsonl"))