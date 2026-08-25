# RefCOCO 本地验证集使用说明

## 用途
本地小验证集，用于快速评估指代表达定位模型，不参与训练。本任务不训练模型，仅做数据校验和本地ACC@0.5测试。

## 环境依赖
```bash
pip install opencv-python numpy jsonlines
