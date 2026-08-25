import json
import random
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np

# ===================== 配置区 =====================
# 原始refcoco转换后的val文件（后面拿到真实数据改这里路径）
INPUT_JSONL = Path("../external_data/refcoco/val.jsonl")
# 输出小验证集
OUTPUT_JSONL = Path("../external_data/refcoco/small_val_400.jsonl")
# 抽取样本数量
SAMPLE_NUM = 400
# 可视化输出目录
VIS_OUT_DIR = Path("../output/small_val_vis")
VIS_SAMPLE_COUNT = 20  # 随机可视化20条
# =================================================

def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

def save_jsonl(samples: List[Dict], path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def check_bbox(sample: Dict):
    """校验bbox：[x1,y1,x2,y2]，归一化0~1，x2>x1,y2>y1"""
    bbox = np.array(sample["bbox"], dtype=np.float32)
    assert bbox.shape == (4,), f"bbox shape error {sample['sample_id']}"
    x1, y1, x2, y2 = bbox
    assert 0.0 <= x1 < x2 <= 1.0, f"x坐标非法 {sample['sample_id']} {bbox}"
    assert 0.0 <= y1 < y2 <= 1.0, f"y坐标非法 {sample['sample_id']} {bbox}"

def check_sample(sample: Dict):
    assert "sample_id" in sample
    assert "query" in sample and len(str(sample["query"]).strip()) > 0, f"空query {sample}"
    assert "image_path" in sample
    assert "bbox" in sample
    check_bbox(sample)

def visualize_one(sample: Dict, out_dir: Path):
    """画框可视化（如果图片不存在就跳过绘图，只打印信息）"""
    img_p = Path(sample["image_path"])
    if not img_p.exists():
        return
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = sample["bbox"]
    # 转回像素坐标
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"{sample['sample_id']}.jpg"
    cv2.imwrite(str(out_file), img)

def main():
    # 如果原始文件不存在，生成模拟数据用来跑通全部逻辑
    if not INPUT_JSONL.exists():
        print(f"警告：未找到 {INPUT_JSONL}，生成模拟400条样本用于调试脚本")
        mock_samples = []
        for i in range(SAMPLE_NUM):
            x1 = random.uniform(0.0, 0.7)
            y1 = random.uniform(0.0, 0.7)
            x2 = x1 + random.uniform(0.05, 0.3)
            y2 = y1 + random.uniform(0.05, 0.3)
            mock_samples.append({
                "sample_id": f"mock_val_{i:06d}",
                "source": "refcoco",
                "query": f"find the cat {i}",
                "image_path": f"mock_images/img_{i}.jpg",
                "bbox": [round(x1,4), round(y1,4), round(x2,4), round(y2,4)]
            })
        all_samples = mock_samples
    else:
        print(f"加载原始数据 {INPUT_JSONL}")
        all_samples = load_jsonl(INPUT_JSONL)

    print(f"总样本数：{len(all_samples)}")
    random.seed(42)
    selected = random.sample(all_samples, k=SAMPLE_NUM)

    # 逐条校验
    valid_list = []
    for s in selected:
        try:
            check_sample(s)
            valid_list.append(s)
        except Exception as e:
            print(f"丢弃非法样本 {s.get('sample_id')} : {e}")

    print(f"校验通过样本：{len(valid_list)}")
    save_jsonl(valid_list, OUTPUT_JSONL)
    print(f"输出小验证集：{OUTPUT_JSONL}")

    # 随机抽20条可视化
    vis_subset = random.sample(valid_list, k=VIS_SAMPLE_COUNT)
    for s in vis_subset:
        visualize_one(s, VIS_OUT_DIR)
    print(f"尝试可视化{VIS_SAMPLE_COUNT}条，图片存在才会保存到 {VIS_OUT_DIR}")

    # 简单统计bbox宽高分布
    widths = []
    heights = []
    for s in valid_list:
        x1,y1,x2,y2 = s["bbox"]
        widths.append(x2-x1)
        heights.append(y2-y1)
    print("\n==== bbox统计（归一化坐标） ====")
    print(f"宽度 均值:{np.mean(widths):.3f} 最小:{np.min(widths):.3f} 最大:{np.max(widths):.3f}")
    print(f"高度 均值:{np.mean(heights):.3f} 最小:{np.min(heights):.3f} 最大:{np.max(heights):.3f}")
    print("\n脚本执行完成！拿到真实val.jsonl后直接重新运行本脚本即可产出正式小验证集。")

if __name__ == "__main__":
    main()
