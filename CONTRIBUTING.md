# 团队协作规范

## 分支

- `main`：随时可运行的稳定版本
- `feature/data-pipeline`：数据与预处理
- `feature/rgb-baseline`：RGB与文本基线
- `feature/multimodal-fusion`：红外、深度和融合
- `feature/evaluation`：评测、推理和提交

每项工作从最新 `main` 创建短期功能分支，不直接向 `main` 推送。

## 标准流程

1. 拉取最新 `main`
2. 创建 `feature/<简短主题>`
3. 小步提交
4. 运行测试和安全检查
5. 推送分支并创建 Pull Request
6. 至少一名其他队员检查后合并

提交信息示例：

```text
feat(data): add depth validity mask
fix(metric): include IoU equal to 0.5
test(submission): reject modified query fields
docs: document server setup
```

## 合并前检查

```bash
python -m pytest -q
python scripts/check_repo_safety.py
git status --short
```

## 禁止提交

- 官方、示例及测试数据
- 模型权重、缓存、日志和预测结果
- SSH私钥、GitHub令牌、平台密码和 `.env`
- 大于5MB且未经团队确认的文件

## 共享服务器

每位成员使用独立工作目录：

```text
/root/autodl-tmp/aic_grounding/workspaces/member_a
/root/autodl-tmp/aic_grounding/workspaces/member_b
/root/autodl-tmp/aic_grounding/workspaces/member_c
```

数据、权重和输出位于工作区外的共享目录。每位成员使用自己的 GitHub 账号、提交姓名和邮箱。
