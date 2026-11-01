# SmartDoc-RAG 运行目录

完整项目介绍、架构、功能边界、知识库格式及真实测试证据统一维护于[仓库主 README](../README.md)。本文件提供在 Code 目录下的运行入口。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env 填写自己的 API Key
python -m streamlit run app.py
```

默认使用 `deepseek-flash` 在线生成与本地 TF-IDF 字符检索；不支持离线大语言模型。本地 TF-IDF 不消耗 Embedding API，在线模式必须配置支持独立 Embedding 的服务，失败不自动回退。界面显示已建知识库的实际模式。

```powershell
# Mock/本地测试，不允许真实 HTTP
python -m pytest tests -v
# 默认不联网的 10 题检索验证
python tests/benchmark/run_eval.py
# 显式真实联调，OPENAI_API_KEY 需预先配置；会产生费用
python tests/benchmark/run_live_api.py --live --evidence ../docs/live-evidence-new.json
```

依赖清单见 [requirements.txt](requirements.txt)，变量模板见 [.env.example](.env.example)。运行数据在 `data/`，凭据与数据均被 Git 忽略。旧 pickle 索引须重新上传构建，原文件不会删除。

[应用说明](../output/SmartDoc-RAG智能应用说明.md) · [真实 API 测试报告](../output/LIVE_API_TEST_REPORT.md)
