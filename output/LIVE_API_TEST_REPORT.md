# SmartDoc-RAG 真实 API 测试报告

执行日期：2026-11-01（Asia/Shanghai）。本报告区分真实调用、Mock 和未测试能力，不使用历史通过记录替代实测证据。

## API 调研与配置

| 项目 | 核实与实测结果 |
|---|---|
| 服务商 | DeepSeek；用户凭据文件仅含 Key，依据既有默认配置和历史来源选择该官方服务，并经真实鉴权确认 |
| 官方文档 | https://api-docs.deepseek.com/ |
| Base URL | https://api.deepseek.com（应用配置示例沿用兼容路径 /v1） |
| 请求/鉴权 | POST /chat/completions；Authorization: Bearer；JSON messages、model、stream；返回 choices/message 或 SSE delta |
| 实际模型 | deepseek-flash；GET /models 实际返回 deepseek-flash、deepseek-v4-pro |
| 选择理由 | 官方较低成本模型，OpenAI 兼容；实际中文 RAG 与流式验证可用，不作未经对比的速度优胜判断 |
| 流式 | 支持 SSE；末尾 data: [DONE]；最终复核 finish_reason=stop |
| Embedding | 官方 API 文档未提供独立 Embedding 端点，不猜测或尝试调用；本地 TF-IDF |
| 成本控制 | 输出最多256 Token，45秒超时，自动重试0，thinking=disabled，Top-K=3；无充值、购买或压力测试 |

配置使用 OPENAI_API_KEY、OPENAI_BASE_URL、MODEL_NAME；密钥仅从用户指定文件读入进程内存，不复制进仓库。可选在线 Embedding 独立配置 EMBEDDING_API_KEY、EMBEDDING_BASE_URL、EMBEDDING_MODEL_NAME。

官方定价页面在引用日期列出 Flash 每百万 Token 输入缓存命中 $0.003–0.006、输入未命中 $0.15–0.30、输出 $0.60–1.20（取决于时段）；未据此假定账户有免费额度。限流文档列出 Flash 账户并发上限2500、超过返回429。串行小样本没有测试限流上限，具体账户额度/账单未查询。

来源名称：DeepSeek 官方首次调用、Chat API、模型/价格、限流、思考模式及模型列表。
链接：https://api-docs.deepseek.com/ 、https://api-docs.deepseek.com/api/create-chat-completion/ 、https://api-docs.deepseek.com/quick_start/pricing/ 、https://api-docs.deepseek.com/quick_start/rate_limit/ 、https://api-docs.deepseek.com/guides/thinking_mode/ 、https://api-docs.deepseek.com/api/list-models 。引用日期：2026-11-01。

## 实际测试结果

| 测试项目 | 测试方式 | 结果 | 证据/备注 |
|---|---|---|---|
| API 连通性/有效鉴权 | GET /models 及真实 Chat | PASS | HTTP 200；模型响应非空 |
| 无效鉴权 | 人工无效测试 Key | PASS | 预期401，未记录完整 Key |
| 中文问答 | 真实模型 | PASS | 返回中文问候，非占位文本 |
| 技术文档直接事实 | 真正上传/解析/切分/建库/恢复后 RAG | PASS | 回答测试文档端口7219，来源live_manual.md |
| 多轮上下文指代 | 使用真实上一轮回答的历史 | PASS | 正确回答“它”的请求超时37秒 |
| 同义改写 | 真实 RAG | PASS | 正确回答03:17执行归档、保留17天 |
| 跨段落问题 | 真实 RAG | PASS | 同时回答端口7219及LQ-417与权限排查 |
| 文档外无关问题 | 真实模型 | PASS | 明确资料未提及外星飞船，未提供不相关来源 |
| 不存在信息 | Streamlit AppTest + 真实模型 | PASS | 对采购预算明确拒答，真实回答渲染成功 |
| SSE/结束标记 | 修正后的一次真实 UI 复核 | PASS | stream=true，45个事件，[DONE]，finish_reason=stop，来源正确 |
| 本地 TF-IDF | 实际构建、查询、存盘、恢复 | PASS | 非在线向量测试 |
| 在线 Embedding | 无独立官方接口 | N/A / NOT TESTED | 未进行真实在线 Embedding 调用 |
| 超时/中断回滚 | 普通 Mock 测试 | PASS（Mock） | 未主动制造真实超时或限流费用 |

独立文档为 Code/tests/benchmark/live_manual.md，站点、7219、37秒、03:17、17天、LQ-417 均明确标为虚构测试数据。真实回答与来源吻合，说明这些案例依据上传文本回答；不能据此证明任意问题都不会使用模型自有知识或产生幻觉。

## 请求账目与观测修复

- 首轮8次Chat（含1次401负例），外加1次模型列表，共9次HTTP。
- 第一次定向流式复核增加1次Chat，累计10次HTTP。此前首轮流式夹具误用 streaming=False；第一次复核位于压缩前字节层，未能识别 SSE，均按原始 FAIL 保留。
- 用户明确回复“不限上限继续”后，只增加1次修正后流式Chat。最终共 **11次HTTP = 10次Chat请求 + 1次模型列表**；Chat包含9次成功生成和1次预期401。
- 修正后观测位于 httpx 自动解压的 iter_bytes 层。实际服务返回 zstd 压缩，最终捕获45个事件、usage和[DONE]；另有 gzip 离线回归。
- 最终业务覆盖项均通过；不将保留的失败记录改为PASS。SSE实际UI复核耗时2.687秒（含AppTest运行，不是纯模型延迟）。首轮各案例耗时0.075–3.436秒，详见JSON，不能外推性能。

## 时间、Token 与证据

首次真实模型列表：2026-11-01T10:53:31.975659+08:00；首轮真实案例10:55:00–10:55:11；最终复核10:59:24开始。

成功捕获用量的7次Chat合计：输入3065、输出276、总计3341 Token。另2次成功流式请求在旧观测器阶段未捕获usage，因此3341不是全轮总消耗；不估造遗漏用量或实际账单金额。

- [模型列表证据](../docs/20261101-live-discovery.json)
- [首轮逐项回答、来源、HTTP及耗时](../docs/20261101-live-evidence.json)
- [第一次流式观测失败](../docs/20261101-stream-recheck.json)
- [最终流式与真实界面复核](../docs/20261101-stream-recheck-v2.json)
- [环境及凭据/原图检查](../docs/20261101-environment-and-safety.json)

未进行独立浏览器人工点选及视觉截图验证；Streamlit 上传控件由 AppTest 注入文件对象，后续保存、解析、建库、LCEL、真实模型、流式和页面消息执行实际应用路径。未查询账户余额、账单，未测试在线 Embedding、真实429/超时及负载上限。

## 复现

在 Code 目录设置自己的 OPENAI_API_KEY，运行：

```powershell
python tests/benchmark/run_live_api.py --live --evidence ../docs/live-evidence-new.json
```

默认为本地 TF-IDF + DeepSeek Flash，脚本最多8次HTTP、关闭自动重试，输出文件必须为新路径。服务商或模型变化后应重新核对官方文档。不得把 Mock 测试当成真实 API 测试。
