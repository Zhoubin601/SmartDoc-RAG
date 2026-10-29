# 第七章 智能应用开发 —— SmartDoc-RAG 智能应用 README 说明文档

- **作业名称**：第七章 智能应用开发
- **满分分值**：100 分
- **智能应用名称**：SmartDoc-RAG 智能文档知识库检索问答系统
- **核心框架**：LangChain (基于 LCEL 链式架构)
- **大模型选型**：在线大模型（兼容 OpenAI 规范，支持 DeepSeek、通义千问等）
- **开源仓库地址**：[https://github.com/Zhoubin601/SmartDoc-RAG](https://github.com/Zhoubin601/SmartDoc-RAG)
- **交付日期**：2026年10月29日

---

## 一、项目设计背景与目标

在大语言模型（LLM）的实际应用中，通用大模型由于知识库切断时间或未接触用户私有资料，往往容易出现“幻觉”现象，无法准确回答特定专业领域或私有文档中的问题。

针对《开源软件构建技术》第七章“智能应用开发”的实践要求，本项目基于行业领先的 **LangChain** 框架，设计并开发了一款具备完整生命周期的多源文档智能知识库检索问答系统 —— **SmartDoc-RAG**。该系统将检索增强生成（RAG）技术与在线大模型相结合，支持解析多格式文档（PDF、DOCX、Markdown、TXT），通过高精度分块切片、自适应向量检索与答案溯源机制，构建起一套企业级规范的文档智能问答解决方案。

---

## 二、系统总体架构与实现原理

系统完全遵循 LangChain 最新的 LCEL（LangChain Expression Language）标准设计，主要由四大核心模块构成：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SmartDoc-RAG 系统架构                          │
├──────────────────┬──────────────────┬──────────────────┬───────────────┤
│ 1. 文档解析模块   │ 2. 向量存储模块   │ 3. LCEL 检索推理  │ 4. Web 交互UI │
│ (Document Loader)│ (Vector Store)   │ (RAG Chain)      │ (Streamlit)   │
├──────────────────┼──────────────────┼──────────────────┼───────────────┤
│ • PDF 多页解析    │ • 自适应向量引擎 │ • ChatPrompt     │ • Swiss 极简   │
│ • Word (.docx)   │ • 余弦相似度计算 │ • LCEL 管道编排  │ • Atkinson 字 │
│ • Markdown / TXT │ • 零额外依赖持久 │ • 知识源格式化   │ • 来源溯源卡片 │
│ • 递归滑动窗口切分│   化(json/npy)   │ • 流式输出响应   │ • 参数实时微调 │
└──────────────────┴──────────────────┴──────────────────┴───────────────┘
```

### 1. 数据处理流程（Ingestion Pipeline）
1. **多格式文件加载**：针对上传的 `.pdf`、`.docx`、`.md`、`.txt` 文件，通过适配解析器提取文本正文，并自动注入元数据（文件名、页码/段落编号）。
2. **递归语义切块**：利用 `RecursiveCharacterTextSplitter` 进行滑动窗口切分，设置合理的 `chunk_size` 与 `chunk_overlap`，保持段落语义连贯。
3. **向量化与持久化**：计算文本向量并持久化存储于本地，实现秒级加载与持久保存。

### 2. 问答推理与溯源（RAG Chain Pipeline）
1. **语义相似度召回**：接收用户提问，计算余弦相似度并召回 Top-K 相关文本切片。
2. **上下文动态注入**：通过 `ChatPromptTemplate` 构建系统提示词，将命中的参考片段注入 Prompt。
3. **在线大模型推理**：调用在线大模型 API 进行深度推理，严格依据参考内容流式生成回答。
4. **知识源显式溯源**：在回答下方渲染引用折叠卡片，标注来源文件、页码与相似度得分。

---

## 三、项目核心技术亮点

1. **原汁原味现代 LangChain LCEL 规范**：
   摒弃了旧版本的黑盒封装，全面采用现代 LCEL 声明式语法构建核心问答链：
   ```python
   chain = (
       prompt_template
       | llm
       | StrOutputParser()
   )
   ```
2. **自适应向量检索技术（破解 DeepSeek API 无 Embedding 痛点）**：
   国内大模型中 DeepSeek 性能优越但官方目前未开放独立 Embedding 接口。本项目创新实现了自适应本地轻量向量检索适配器（`SmartVectorStore`），完全继承自 LangChain 官方 `VectorStore` 规范，免去了下载数 GB 权重和安装大型数据库的繁琐，实现秒级冷启动与零成本高效检索。
3. **遵循 ui-ux-pro-max 专业人机交互设计**：
   前端基于 Streamlit 开发，界面设计遵循 **Swiss Minimalism（瑞士极简风格）** 与 **Atkinson Hyperlegible** 无障碍排版体系，全站符合 WCAG AA 4.5:1 对比度标准，提供流式打字机视觉交互与精美的来源引用卡片。

---

## 四、工程目录结构

```text
Code/
├── .env.example              # 环境变量配置模板
├── requirements.txt          # 项目依赖清单
├── config.py                 # 全局参数与配置加载
├── app.py                    # Streamlit Web 可视化交互主入口
├── src/                      # 核心业务逻辑模块
│   ├── __init__.py
│   ├── models.py             # 在线大模型与 Embedding 工厂适配器
│   ├── document_loader.py    # 多格式文档解析与语义切分器
│   ├── vector_store.py       # 兼容 LangChain 标准的自适应向量存储
│   └── rag_engine.py         # 端到端 LCEL 检索增强问答引擎
└── data/                     # 运行时数据与知识库持久化目录
    ├── uploads/              # 用户上传的原始文档
    └── vector_db/            # 持久化的向量与文档元数据
```

---

## 五、环境安装与快速启动步骤

### 1. 安装依赖
```bash
cd Code
pip install -r requirements.txt
```

### 2. 环境变量配置
复制配置模板：
```bash
cp .env.example .env
```
根据实际情况填写 API 参数（例如 DeepSeek API）：
```env
OPENAI_API_KEY=sk-your-deepseek-api-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
TEMPERATURE=0.3
```

### 3. 启动交互界面
```bash
streamlit run app.py
```
启动后在浏览器中访问 `http://localhost:8501` 即可体验。

---

## 六、开源仓库与源码

- **开源仓库地址**：[https://github.com/Zhoubin601/SmartDoc-RAG](https://github.com/Zhoubin601/SmartDoc-RAG)
- **开源协议**：MIT License
