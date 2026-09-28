# Round 1：服务器只读资产审计

这个执行包必须在复赛 Linux 服务器上运行。它检查服务器上的仓库、数据、GPU、Python、GroundingDINO 源码、配置和候选权重；它不会检查当前 Windows 电脑。

## 安全边界

脚本不会：

- 安装或卸载软件；
- 访问网络；
- 修改、提交、切换或推送 Git；
- 修改官方数据、源码、配置或权重；
- 复制模型权重；
- 输出完整环境变量、令牌、密钥或 SSH 配置。

脚本只会在 `${TMPDIR:-/tmp}` 下创建审计结果目录和一个 `.tar.gz` 结果包。

## 运行前确认

执行包在服务器上应位于：

```text
/srv/nfs/home/njnu_lhf/aic-repechage/project/server_packets/round_1_audit/
```

先登录服务器，再执行：

```bash
cd /srv/nfs/home/njnu_lhf/aic-repechage/project
bash server_packets/round_1_audit/run.sh
```

无需激活 Conda 环境。脚本会记录当前默认 Python，也会列出可发现的 Conda 环境，供 Codex 下一轮分析。

## 运行时间

通常约 1～5 分钟。候选 Swin-B 权重需要计算 SHA-256；权重位于较慢的 NFS 时可能更久。

## 运行完成后

终端最后会打印：

```text
AUDIT_RESULT_DIR=...
AUDIT_ARCHIVE=...
```

请把 `AUDIT_ARCHIVE` 指向的 `.tar.gz` 文件带回本地并交给 Codex。无需复制数据、权重或整个服务器仓库。

如果没有生成压缩包，请把 `AUDIT_RESULT_DIR` 整个目录带回。至少应包含 `STATUS.txt`、`summary.json`、`stdout.log` 和 `stderr.log`。

## 可选输出位置

默认输出到 `/tmp`。如服务器策略不允许，可显式指定一个仓库外目录：

```bash
bash server_packets/round_1_audit/run.sh --output-root /srv/nfs/home/njnu_lhf/aic-repechage/audit_results
```

