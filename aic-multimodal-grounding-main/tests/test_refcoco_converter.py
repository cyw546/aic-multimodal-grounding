import json
from pathlib import Path
import random

# 直接复制函数过来，不import scripts，绕开refer依赖
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

    print("\n====随机抽查20条样本====")
    pick = random.sample(samples, k=20)
    for p in pick:
        print(f"sample_id:{p['sample_id']}, query:{p['query']}, bbox:{p['bbox']}")

    if ok:
        print("\n✅全部校验通过")
    else:
        raise AssertionError("存在非法样本")
    return samples


def test_bbox_logic():
    """mock测试bbox校验函数，不需要真实数据集，不需要refer库"""
    ok1, _ = is_bbox_valid([10,20,100,120], 640, 480)
    assert ok1 == True

    ok2, _ = is_bbox_valid([50,50,30,80],640,480)
    assert ok2 == False

    ok3, _ = is_bbox_valid([-2,10,100,100],640,480)
    assert ok3 == False
    print("✅ bbox校验单元测试通过")


if __name__ == "__main__":
    test_bbox_logic()
    # 拿到真实jsonl文件之后再取消注释
    # check_jsonl(Path("./output_refcoco/train.jsonl"))
    # check_jsonl(Path("./output_refcoco/val.jsonl"))
