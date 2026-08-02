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
