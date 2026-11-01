# SmartDoc-RAG 智能应用说明

SmartDoc-RAG 用于对上传的文档进行检索和问答。应用采用 LangChain LCEL 和 Streamlit，先从文档中找出相关片段，再调用在线大语言模型生成回答，并显示对应来源。默认检索方式为本地 TF-IDF，回答生成需要在线模型服务。

课程 开源软件构建技术 第七章 智能应用开发

开源仓库：[Zhoubin601/SmartDoc-RAG](https://github.com/Zhoubin601/SmartDoc-RAG)

开源许可 MIT

项目说明：[仓库 README](../README.md)

## 功能与实现

处理流程：上传文档 → 保存和解析 → 递归字符切分 → 向量化 → Top-K 检索 → 在线模型生成回答 → 显示来源。

| 模块 | 当前实现 |
| --- | --- |
| 文档处理 | PDF 文字层、DOCX 正文段落、Markdown、TXT；逐文件报告解析错误 |
| 切分与检索 | RecursiveCharacterTextSplitter 字符切分；NumPy 余弦相似度 Top-K |
| 模型与问答 | ChatPromptTemplate、ChatOpenAI、StrOutputParser；多轮历史与流式输出 |
| 来源与安全 | 转义文件名、页码及正文；UUID 内部文件名；API 错误脱敏 |
| 知识库 | JSON 状态保存与恢复；add_documents() 接口支持增量添加 |

本地 TF-IDF 提取长度为 2 和 3 的字符片段，最多保留 2048 个特征，按字面重合程度检索，不具备深度语义向量模型的理解能力。在线 Embedding 需要另行配置支持向量接口的服务；配置或调用失败时，页面会报告错误，不自动切换为本地模式。侧边栏显示已建知识库实际使用的检索方式。

模型最多接收最近 6 条有效用户和助手消息，用于理解追问中的指代。检索结果的最高分低于 0.08 时，提示词要求模型说明资料不足；最高分达到 0.08 但低于 0.15 时，会加入低相关度提醒。分数表示检索相似度，不是回答正确率；回答仍需结合原文核对。

## 安装配置与使用

先安装 Python，再在仓库的 Code 目录安装依赖并启动应用。以下命令适用于 Windows PowerShell。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run app.py
```

将 .env 中的 API Key 替换为自己的密钥，也可以在页面侧边栏填写。配置示例如下：

```dotenv
OPENAI_API_KEY=replace-with-your-own-key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-flash
TEMPERATURE=0.3
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=4
```

选择本地 TF-IDF 时，不需要 Embedding 密钥。选择在线 Embedding 时，另行填写 EMBEDDING_API_KEY、EMBEDDING_BASE_URL 和 EMBEDDING_MODEL_NAME，且所用服务需要提供独立的向量接口。已保存的 DeepSeek 问答测试采用本地 TF-IDF 检索和在线模型生成。

上传文档后，点击“解析并构建向量知识库”，即可提问、追问和查看引用。该按钮会重新构建知识库；增量添加由 add_documents() 代码接口提供。重启后，选择与建库时一致的检索方式、模型和接口地址，即可恢复已保存的知识库。文档、向量矩阵、模型信息及 TF-IDF 词表和 IDF 权重保存在快照目录中，CURRENT 文件指向有效快照。“清空知识库与重置对话”只清除当前会话，不删除磁盘文件。

## 验证结果

在 Code 目录运行：

```powershell
python -m pytest tests -v
python tests/benchmark/run_eval.py
# 真实联调需预先配置自己的 OPENAI_API_KEY，会产生费用
python tests/benchmark/run_live_api.py --live --evidence ../docs/live-new.json
```

保存的 pytest 执行记录显示：94 项测试通过，0 项失败，0 项跳过。测试涉及文档解析、空文件和损坏文件、上传文件重名、HTML 转义、密钥脱敏、Embedding 配置及异常、知识库增量与恢复、多轮历史、拒答提示、流式异常和页面状态。普通单元测试使用本地数据或模拟模型，不调用真实 API。

保存的 DeepSeek Flash 实际调用记录覆盖中文问答、文档事实、同义改写、跨段落问题、多轮指代，以及无关和缺失信息的拒答。在一次流式请求中记录到 45 个 SSE 事件及 [DONE] 结束标记，页面完整显示回答并保留来源。界面流程由 Streamlit AppTest 验证，没有人工浏览器验收记录。

自动化测试结果见[执行记录](../docs/20261101-pytest-final.xml)；真实接口的请求、结果和已知限制见 [API 测试报告](LIVE_API_TEST_REPORT.md)。

## 使用限制

PDF 需要包含文字层，扫描件应先进行 OCR。DOCX 只读取正文段落，不提取表格文字，也不能还原可靠的原始页码。本地 TF-IDF 依赖字符重合，问题用词与原文相差较大时，可能找不到合适片段。在线 Embedding 仅有模拟和兼容性测试记录，尚未完成真实服务验证。

应用面向本地课程演示，尚未实现多用户鉴权和租户隔离。知识库快照不会自动清理。现有测试不包含压力测试、无障碍合规评估或完整的跨平台验证，也没有全新虚拟环境安装记录。小型文档测试的结果不能推广为任意资料上的回答准确率。

项目源码位于 Code，使用说明见仓库根 README.md，测试与分析记录位于 docs。应用没有接入离线大语言模型，生成回答时需要可访问的在线模型服务。
