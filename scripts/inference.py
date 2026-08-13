import json
import os

from src.data.dataset import GroundingDataset
from baseline.inference import Baseline

# 数据路径
DATA_ROOT = (
    "/root/autodl-tmp/aic_grounding/"
    "data/official/preliminary"
)

ANNOTATION_FILE = (
    "/root/autodl-tmp/aic_grounding/data/official/preliminary/queries/queries.json"
)

# GroundingDINO配置
CONFIG_PATH = (
    "/root/autodl-tmp/aic_grounding/workspaces/member_a/aic-multimodal-grounding/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
)

WEIGHT_PATH = (
    "/root/autodl-tmp/aic_grounding/weights/groundingdino_swinb_cogcoor.pth")

OUTPUT_FILE = "outputs/predictions2.json"

def main():
    # 1. 加载数据集
    dataset = GroundingDataset(
        DATA_ROOT,
        ANNOTATION_FILE
    )

    # 2. 加载模型
    baseline = Baseline(
        CONFIG_PATH,
        WEIGHT_PATH
    )

    results = {}

    # 3. 遍历数据
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample["id"]
        print(
                    f"[{i+1}/{len(dataset)}]",
                    sample_id
                )
        prediction = baseline.predict(
            sample["visible"],
            sample["query"]
        )
        if prediction is None:
            print(
                "No prediction:",
                sample["id"],
                sample["query"]
            )
            bbox = [0,0,0,0]
        
        else:
            bbox = prediction["bbox"]
                
        results[sample["id"]] = {
            "visible": sample["visible"],
            "infrared": sample["infrared"],
            "depth": sample["depth"],
            "query": sample["query"],
            "bbox": bbox
        }
    
    # 6. 保存json
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )
    print(
        "saved:",
        OUTPUT_FILE
    )

if __name__ == "__main__":

    main()
