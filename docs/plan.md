# 执行计划与推进路线

## 阶段一：需求确认与方案定型（已完成）
- [x] 解析原始需求截图 `raw/要求.png`。
- [x] 梳理技术约束并建立过程文档（`docs/source-notes.md`, `docs/analysis-notes.md`）。
- [x] 用户确认核心应用方向为 SmartDoc-RAG，模型选型为在线大模型 API，仓库链接为 `https://github.com/Zhoubin601/SmartDoc-RAG`，代码放置于 `Code/` 目录下。

## 阶段二：工程脚手架与核心代码开发（已完成）
- [x] 引入 `ui-ux-pro-max` 技能，完成 Swiss Minimalism 风格、Atkinson Hyperlegible 字系与 WCAG 4.5:1 设计规范。
- [x] 创建基础配置与环境变量管理：`Code/.env.example`、`Code/config.py`、`Code/requirements.txt`。
- [x] 封装模型适配工厂（`Code/src/models.py`）：无缝支持在线 API（兼容 OpenAI 规范，如 DeepSeek、通义千问）与轻量自适应本地向量检索保底。
- [x] 编写文档解析与分块模块（`Code/src/document_loader.py`）：支持 PDF、DOCX、Markdown、TXT。
- [x] 编写高精度自适应本地向量检索库（`Code/src/vector_store.py`）：完全兼容 LangChain `VectorStore` 抽象规范，支持余弦相似度打分与本地持久化。
- [x] 构建 LangChain LCEL RAG 核心链（`Code/src/rag_engine.py`）：实现向量持久化、检索器、提示词模板及问答链。
- [x] 开发交互式前端（`Code/app.py`）：基于 Streamlit 实现优雅的 Web 界面，包含文档上传、向量库构建、流式问答与源文档溯源。

## 阶段三：测试与功能验证（已完成）
- [x] 本地代码语法与模块依赖导入检查（全部通过）。
- [x] 验证向量持久化与相似度检索单元测试（全部通过）。

## 阶段四：正式交付物输出（已完成，输出至 `output/`）
- [x] 编写项目开源专属说明文档 `Code/README.md`。
- [x] 编写规范交付文档 `output/20261029-smartdoc-rag-README说明文档-v1.md`。
- [x] 编译生成课程平台标准提交文档 `output/20261029-smartdoc-rag-README说明文档-v1.docx`。
- [x] 配套开源项目基础设施：`Code/.gitignore` 与 `Code/LICENSE`。
