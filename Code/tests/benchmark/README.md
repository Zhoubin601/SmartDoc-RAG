# 评测夹具与运行说明

`sample_manual.md` 是既有 10 题测试的虚构技术手册，其中 PostgreSQL、RPC、Prometheus、`rebuild_index` 命令等仅为检索测试内容，并非本应用实际功能或部署要求。`live_manual.md` 同样是明确标注的虚构站点手册，用于检查模型是否按上传文本回答特殊数值。

- `eval_questions.json`：既有 10 题，普通 pytest 只验证本地检索、提示词及 Mock 行为。
- `run_eval.py`：默认不联网；只有 `--live` 才读取环境中的 Key 并调用真实模型。报告不覆盖已有文件，Mock 生成列标为 NOT TESTED。
- `run_live_api.py`：必须显式 `--live`，默认最多 8 次 HTTP（含无效 Key 负例），输出上限 256，自动重试为 0；验证本地 TF-IDF 与真实在线 Chat，不测试在线 Embedding。使用前参阅[主 README](../../../README.md)。
- `BENCHMARK_REPORT.md`：2026-10-29 历史报告。本轮发现旧评测曾把低检索分当成模型拒答通过，原统计未重算，不能作为当前质量或普遍准确率的证据；当前验证以 [LIVE_API_TEST_REPORT.md](../../../output/LIVE_API_TEST_REPORT.md) 为准。
