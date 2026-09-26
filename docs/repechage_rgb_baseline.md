# 复赛 RGB GroundingDINO 基线

## 结论

V1 推荐参数：

```yaml
model:
  box_threshold: 0.25
  text_threshold: 0.25
  fallback_box_threshold: 0.05
  fallback_text_threshold: 0.05
```

固定的前 100 条复赛查询在四组配置下均生成了合法的 `[0, 1]` `xyxy_norm` 框，最终 100 个预测框和分数逐项完全一致。`0.25/0.25` 是唯一无需回退推理即可覆盖 100/100 条查询的配置，并且平均耗时最低，因此选为 V1。

本结论只比较输出完整性、候选覆盖率、置信分数、耗时和人工可视化；官方复赛数据没有在本实验中提供真实框，因此不能把置信分数解释为定位准确率。

## 固定环境

- 项目目录：`/srv/nfs/home/njnu_lhf/aic-repechage/project`
- 数据目录：`/srv/nfs/data/aic-repechage/official/repechage`
- Python：3.10.21
- PyTorch：2.3.1+cu121
- torchvision：0.18.1+cu121
- CUDA runtime：12.1
- GPU：NVIDIA RTX A6000
- GroundingDINO 源码：`https://github.com/IDEA-Research/GroundingDINO`
- GroundingDINO 提交：`856dde20aee659246248e20734ef9ba5214f5e44`
- 模型配置：`groundingdino/config/GroundingDINO_SwinB_cfg.py`
- 权重文件：`groundingdino_swinb_cogcoor.pth`
- 官方权重地址：`https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth`
- 权重大小：938057991 字节
- 权重 SHA-256：`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`

服务器无法稳定访问 Hugging Face 的 Xet 存储。官方 Swin-B checkpoint 已包含 BERT 的 embeddings、12 层 encoder 和 pooler 参数，因此执行包对固定 GroundingDINO 提交应用了离线加载补丁：tokenizer 从服务器缓存读取，BERT 按标准 `BertConfig` 构造，再由官方 checkpoint 覆盖训练参数。补丁 SHA-256 为 `f3c669254735c1ed5231de8a8c4d6515bf3300775e2d3d6f7c62eace4063d65b`。

## 实验方法

数据集共有 5690 条查询。实验固定使用 `queries/queries.json` 当前顺序的前 100 个 Query ID；查询清单保存在结果中的 `query_ids_100.json`。四组配置共用：

- fallback box threshold：0.05
- fallback text threshold：0.05
- 相同模型、权重、查询顺序和 RGB 输入
- 主阈值无候选时才执行一次回退推理
- 最终结果通过仓库 JSON 校验器验证

先验证 1 条和 10 条，再运行完整 100 条。服务器命令为：

```bash
/srv/nfs/home/njnu_lhf/aic-repechage/env/bin/python \
  scripts/run_repechage_rgb_experiment.py \
  --data-root /srv/nfs/data/aic-repechage/official/repechage \
  --model-config /srv/nfs/home/njnu_lhf/aic-repechage/external/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
  --weights /srv/nfs/home/njnu_lhf/aic-repechage/weights/groundingdino_swinb_cogcoor.pth \
  --groundingdino-root /srv/nfs/home/njnu_lhf/aic-repechage/external/GroundingDINO \
  --output-root /srv/nfs/home/njnu_lhf/aic-repechage/outputs/rgb_baseline_v1_20260926T111303Z \
  --max-samples 100 \
  --visualization-count 20 \
  --resume
```

最终运行时间为 150.09 秒，其中包括模型加载、四组推理、校验和可视化。测试结果为 61 passed、2 skipped。

## 阈值对比

| box | text | 主阈值成功 | 主阈值无候选 | 回退成功 | 默认框 | 平均分 | 平均耗时/条 | 总推理耗时 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.25 | 89 | 11 | 11 | 0 | 0.537766 | 0.273854 s | 27.3854 s |
| 0.25 | 0.25 | 100 | 0 | 0 | 0 | 0.537766 | 0.244849 s | 24.4849 s |
| 0.35 | 0.20 | 89 | 11 | 11 | 0 | 0.537766 | 0.269960 s | 26.9960 s |
| 0.30 | 0.20 | 98 | 2 | 2 | 0 | 0.537766 | 0.247616 s | 24.7616 s |

补充统计：分数中位数 0.530053，最小值 0.272857，最大值 0.905620。所有配置均为 100/100 合法框且没有默认框。相对 `0.35/0.25`，推荐配置的平均单条耗时降低约 10.6%。

`0.35/0.25` 和 `0.35/0.20` 的无候选 Query ID 完全相同，说明这批样本中把 text threshold 从 0.25 降至 0.20 没有改善主阶段覆盖率。`0.30/0.20` 仍有 2 条需要回退：`000029_003` 和 `000211_004`。

## 人工检查

已逐张检查结果包中的 20 张四配置对比图。四组阈值在这些样本上画出的最终框相同，颜色差异只表示结果来自主阈值还是回退阈值。

观察到的代表性情况：

- 定位较清楚：路灯、花盆、石阶、蹲下的人、靠近镜头的女性、建筑上的单个蓝色汉字。
- 关系词可能失效：`000001_003` 要求“从左到右第二把伞”，结果选择了左侧第一把大伞。
- 小目标会退化为较大主体框：`000015_002` 查询牛仔裤后袋中的手机，结果框住整个人。
- 文字和符号区分较弱：`000113_002/004/005` 对指定字母发生混淆；`000211_006` 查询玻璃中的倒影，但结果选择了实体字母。
- 语义误检仍然存在：`000022_004` 的减速带和 `000029_003` 的橙色路牌都被定位到大型红色装置。
- 组合目标常得到较宽框：`000211_001` 和 `000211_004` 的框覆盖了目标字母及邻近字母。

因此，降低阈值解决的是“无候选”和重复推理问题，不会修复空间关系、小物体、文字辨识或倒影理解等语义错误。后续若继续提升效果，应优先研究查询改写、候选重排或多模态信息，而不是继续微调这四个阈值。

## 输出与安全

- 结果状态：PASS
- 固定 100 条查询全部成功推理
- 4 份 `prediction.json` 均通过提交格式校验
- 4 份 `details.jsonl` 记录 stage、候选数、分数和耗时
- 20 张可视化及 `visualization_manifest.json` 已生成
- 结果包全部文件通过 `checksums.txt` 校验
- 仓库安全检查通过
- 权重、官方数据、预测大文件、环境和第三方源码均未加入 Git
