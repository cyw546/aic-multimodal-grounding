# Round 2：环境恢复与 RGB 阈值实验

本执行包在复赛 Linux 服务器上运行。它会使用 NFS 目录准备专用 Python 环境、固定版本 GroundingDINO 和 Swin-B 权重，然后依次完成 1 条、10 条和固定 100 条的四组阈值实验。

## 会发生的写操作

- 必要时更新 `/srv/nfs/home/njnu_lhf/aic-repechage/env` 中缺失或不兼容的 Python 包；如果该环境的 Python 版本不兼容，则新建 `env_rgb_baseline`，不删除原环境。
- 在 `/srv/nfs/home/njnu_lhf/aic-repechage/external/` 准备固定提交的 GroundingDINO。
- 在 `/srv/nfs/home/njnu_lhf/aic-repechage/weights/` 保存官方 Swin-B 权重。
- 将执行包内三个经过校验的项目文件写入项目目录；覆盖已知旧版文件前会检查 SHA-256，发现未知改动会停止。
- 在 `/srv/nfs/home/njnu_lhf/aic-repechage/outputs/` 保存实验输出、日志和结果包。

脚本不会创建 Git commit、不会 push、不会修改官方数据，也不会把权重或预测文件加入 Git。

## 网络要求

如果源码或权重尚不存在，脚本需要访问：

- `https://github.com/IDEA-Research/GroundingDINO.git`
- GroundingDINO GitHub Release 或官方 Hugging Face 镜像
- PyTorch 和 PyPI 软件包索引
- Hugging Face 上的 `bert-base-uncased` 模型资源

执行包已携带固定提交的 GroundingDINO 源码归档，以减少对 GitHub clone 的依赖。官方 Swin-B 权重约 895 MiB，作为独立文件传输，以避免重复压缩。脚本会优先读取：

```text
/srv/nfs/home/njnu_lhf/aic-repechage/groundingdino_swinb_cogcoor.pth
```

该文件的预期信息为：

```text
大小：938057991 字节
SHA-256：46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7
```

脚本会再次校验 SHA-256；文件不存在时才尝试从官方 GitHub Release 或 Hugging Face 镜像下载。

GroundingDINO checkpoint 已包含完整的 BERT 参数。执行包会对固定提交中的语言模型加载器应用一个经过校验的离线补丁：tokenizer 只读取服务器缓存，BERT 按标准 `BertConfig` 构造后由官方 GroundingDINO checkpoint 覆盖全部训练参数，从而避免模型加载阶段再次下载约 440 MB 的 Hugging Face 权重。

## 推荐运行方式

执行包应解压到仓库外的工作目录，避免源码归档和传输 ZIP 被仓库安全检查误判：

```text
/srv/nfs/home/njnu_lhf/aic-repechage/round_2_experiment/
```

推荐使用：

```bash
cd /srv/nfs/home/njnu_lhf/aic-repechage
tmux new -s repechage_rgb
bash round_2_experiment/run.sh
```

在 `tmux` 中按 `Ctrl-b`，再按 `d` 可退出但保持任务运行。重新连接后使用：

```bash
tmux attach -t repechage_rgb
```

## 断点恢复

如果脚本在推理阶段中断，使用终端打印的同一个实验输出目录：

```bash
bash /srv/nfs/home/njnu_lhf/aic-repechage/round_2_experiment/run.sh \
  --resume-output /srv/nfs/home/njnu_lhf/aic-repechage/outputs/<已有目录名>
```

## 带回结果

结束时终端会打印：

```text
ROUND2_STATUS=PASS 或 FAIL
ROUND2_RESULT_DIR=...
ROUND2_ARCHIVE=...tar.gz
```

只需把 `ROUND2_ARCHIVE` 带回本地。压缩包包含日志、统计、100 条预测明细和至少 20 张可视化，不包含权重、数据集、Python 环境或 GroundingDINO 源码。
