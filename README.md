# AIC Multimodal Visual Grounding

本项目用于“基于大模型的多模态视觉理解与推理”算法挑战赛。输入为可见光、热红外、深度图像和英文查询文本，输出为目标在可见光图像中的归一化边界框。

## 任务与指标

预测格式为 `[x1, y1, x2, y2]`，坐标范围为 `[0, 1]`。主指标为 ACC@0.5：预测框与真实框的 IoU 大于或等于 0.5 时记为正确。

## 当前功能

- 官方 JSON 与 RGB、红外、16位深度图读取
- 三模态空间对齐、路径与标签校验
- IoU、ACC@0.5 和诊断指标
- 预测 JSON 与提交 ZIP 合法性检查
- CPU 单元测试

## 目录结构

```text
configs/                 数据与训练配置
scripts/                 数据检查与仓库安全检查
src/data/                数据读取
src/metrics/             IoU与ACC@0.5
src/submission/          JSON和ZIP提交检查
tests/                   单元测试
```

比赛数据、权重、缓存、日志和预测结果均保存在仓库外，不得上传 GitHub。

## 环境安装

推荐 Ubuntu 22.04、Python 3.10、PyTorch 2.3 和 CUDA 12.1。

```bash
conda env create -f environment.yml
conda activate aic
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## 数据与测试

服务器共享数据目录为 `/root/autodl-tmp/aic_grounding/data`，仓库中不保存数据。

```bash
AIC_SAMPLE_ROOT=/root/autodl-tmp/aic_grounding/data/sample python -m pytest -q
python scripts/check_repo_safety.py
```

## 提交检查

```bash
python -m src.submission.validator \
  --original /path/to/original.json \
  --prediction /path/to/prediction.json

python -m src.submission.validator \
  --original /path/to/original.json \
  --zip /path/to/submission.zip
```


详细协作规则见 `CONTRIBUTING.md`。严禁提交赛事数据、模型权重、预测结果、访问令牌或 SSH 密钥。

# RGB Grounding Baseline

本模块基于 GroundingDINO，实现 RGB 图像与文本 Query 的视觉定位接口。

## 模型与权重

模型：GroundingDINO SwinB

固定使用 GroundingDINO 官方提交：

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout 856dde20aee659246248e20734ef9ba5214f5e44
python -m pip install --no-deps --no-build-isolation -e .
```

服务器权重路径：

```text
/srv/nfs/home/njnu_lhf/aic-repechage/weights/groundingdino_swinb_cogcoor.pth
```

源码、权重和缓存均不得提交到本仓库。

## 推理参数

- box_threshold: 0.25
- text_threshold: 0.25
- fallback_box_threshold: 0.05
- fallback_text_threshold: 0.05

这是固定 100 条复赛查询上四组阈值实验得到的 V1 推荐配置。完整环境、权重校验、结果表与人工检查记录见 [`docs/repechage_rgb_baseline.md`](docs/repechage_rgb_baseline.md)。

## 1. 接口说明

对外接口为：

```python
Baseline.predict(image, query)
```

输入：

- `image`：RGB `numpy.ndarray`
  - shape：`(H, W, 3)`
  - dtype：`uint8`
- `query`：非空字符串

输出格式：

```python
{
    "bbox": [x1, y1, x2, y2],
    "score": float
}
```

其中：

- `bbox` 为归一化 `xyxy` 坐标；
- 坐标范围为 `[0, 1]`；
- `score` 为当前最佳候选框得分；

## 2. 使用方法

```
from baseline.inference import Baseline


baseline = Baseline(
    config_path="<GROUNDING_DINO_CONFIG_PATH>",
    weight_path="<GROUNDING_DINO_WEIGHT_PATH>"
)


result = baseline.predict(
    image=rgb_image,
    query="the silver light bulb"
)


print(result)
```

注意：`rgb_image` 必须是 RGB 顺序的 `numpy.ndarray`。

如果使用 OpenCV 读取图片：

```
import cv2


image = cv2.imread("example.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


result = baseline.predict(
    image,
    "the silver light bulb"
)
```

## 模型评估与候选框诊断

统一评估器、固定验证清单和候选覆盖率说明见 [`docs/model_evaluation_and_candidate_diagnosis.md`](docs/model_evaluation_and_candidate_diagnosis.md)。入口命令为：`python scripts/evaluate_predictions.py --help`。
