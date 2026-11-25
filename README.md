# gemini-codex · v5.1 合成验收包（审查修订）

这是 Liu/LRPHSD 固定部署巡检研究的**工程回归包**，不是作者官方实现、已训练模型、hn01 验收回执或论文效果复现。A 检查固定机位下的物理路线、成本与同风参照解；B 检查在线复检事件执行器；两条线独立验收。研究背景：[Liu, Hu & Cheng · EJOR](https://ideas.repec.org/a/eee/ejores/v334y2026i1p285-300.html)。

## 运行

Python 3.10+（标准库），在本仓库根目录：
```bash
# 只重放随包的人工夹具、独立比较和单元测试。不训练，不生成研究地图。
python3 run_acceptance_pack.py
# 或指定一个尚不存在的目录，保留每步 stdout/stderr、输入输出 SHA 及结果
python3 run_acceptance_pack.py --output-dir runs/check-001
```
有受信的冻结代码副本时可用下式作**源码兼容性检查**（ROOT 是包含 production/frozen 的目录，不是 frozen 自身）。源码不随本包分发。
```bash
python3 run_acceptance_pack.py --frozen-root "$FROZEN_ROOT" --output-dir runs/frozen-001
```
为保留已发命令，`--hn01-repo-root` 是 `--frozen-root` 的别名。主入口先核 `frozen_source_lock.json`，再让 A 调用实际冻结验证器。**提供本地或服务器路径都不证明主机身份**。脚本在任何主机都只报告 `ENGINEERING_PASS` 或 `FAILED`；`hn01_g1_status` 保持 `UNVERIFIED_PENDING_HN01_RECEIPT`。失败目录和日志保留，不覆盖旧回执。后续在 hn01 的签认须由项目负责人核对运行主机、输入身份、预算与原始输出，本包不会自动开训。

## 阅读顺序

- `SPEC_V5_1_ACCEPTANCE.md`：范围、验收、预期结果和仍未获证的主张。
- `track_a/inputs`、`track_a/expected`：手工构造的参照路线及结果，不是 Liu 作者求解输出。
- `track_b/inputs`、`track_b/expected`：预先冻结的 7 条班次和 37 个检查点。盲执行与读取期望的比较器分别运行。
- `tests/`：正控和故意损坏的负控。`stats/` 是统计工具与诊断；不提供新地图、强基线对比或发表依据。
- `evidence/local_engineering.json`：本次发布前两种工程模式的摘要和输入/输出指纹；不是服务器回执。
- `runs/`：本地生成并被 Git 忽略的回执、原始日志与计时；不把旧包自带的输出冒充当前结果。

无需发布脚本即可使用本包。`push_to_github.sh` 仅供维护者在本仓库、干净且已提交的分支上使用：不修改 remote、不替用户暂存文件、不强推。
