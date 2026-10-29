# -*- coding: utf-8 -*-
"""
SmartDoc-RAG 智能文档知识库助手 - Streamlit Web 交互主程序
UI/UX 规范遵循: ui-ux-pro-max (Minimalism & Swiss Style, Atkinson Hyperlegible, WCAG AA 4.5:1)
"""
import os
import sys
from pathlib import Path

# 确保项目根路径进入 sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import streamlit as st
import config
from src.models import get_chat_model, get_embedding_model, mask_secret
from src.document_loader import load_and_split_files
from src.vector_store import SmartVectorStore
from src.rag_engine import RAGEngine

# ==============================================================================
# 页面元信息配置
# ==============================================================================
st.set_page_config(
    page_title="SmartDoc-RAG 智能文档知识库",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 注入基于 UI-UX-PRO-MAX (Swiss Style & Atkinson Hyperlegible) 设计系统样式
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&display=swap');

:root {
  --color-primary: #475569;
  --color-secondary: #64748B;
  --color-accent: #2563EB;
  --color-background: #F8FAFC;
  --color-foreground: #1E293B;
  --color-card: #FFFFFF;
  --color-muted: #EAEFF3;
  --color-muted-foreground: #475569;
  --color-border: #E2E8F0;
  --color-destructive: #DC2626;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 15px rgba(0,0,0,0.08);
}

/* 全局字体与平滑渲染 */
html, body, [class*="css"] {
    font-family: 'Atkinson Hyperlegible', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--color-foreground);
    background-color: var(--color-background);
}

/* 顶部品牌导航栏 */
.header-container {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    background: #FFFFFF;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 24px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #EFF6FF;
    border-radius: 10px;
    color: var(--color-accent);
}

.brand-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--color-foreground);
    margin: 0;
    line-height: 1.2;
}

.brand-subtitle {
    font-size: 0.88rem;
    color: var(--color-secondary);
    margin: 4px 0 0 0;
}

/* 状态标签样式 */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: #F1F5F9;
    color: var(--color-primary);
    border: 1px solid var(--color-border);
}

.status-pill-success {
    background: #ECFDF5;
    color: #047857;
    border-color: #A7F3D0;
}

/* 引用文档卡片 (Citation Card) */
.citation-card {
    background: #FFFFFF;
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-accent);
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 10px;
    box-shadow: var(--shadow-sm);
    transition: all 200ms ease;
}

.citation-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

.citation-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--color-secondary);
    margin-bottom: 8px;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 6px;
}

.score-badge {
    background: #EFF6FF;
    color: var(--color-accent);
    padding: 2px 8px;
    border-radius: 4px;
    font-family: monospace;
}

.citation-body {
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
}

/* 侧边栏及按钮微交互优化 */
button, [role="button"] {
    cursor: pointer !important;
    transition: all 180ms ease !important;
}

button:hover {
    opacity: 0.92;
}

/* 自定义折叠面板与边框美化 */
.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--color-foreground) !important;
    border-radius: 8px !important;
}

/* 禁用 layout-shifting 动画并保持 WCAG 可见焦点 */
:focus-visible {
    outline: 2px solid var(--color-accent) !important;
    outline-offset: 2px !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 会话状态初始化
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ==============================================================================
# 侧边栏控制面板
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ 系统与模型配置")

    # 1. 大语言模型配置
    st.markdown("**1. 在线大模型 (LLM)**")
    api_key = st.text_input(
        "API Key",
        value=config.DEFAULT_OPENAI_API_KEY,
        type="password",
        placeholder="sk-...",
        help="支持 DeepSeek、通义千问、OpenAI、硅基流动等所有兼容 OpenAI 规范的 API Key"
    )
    base_url = st.text_input(
        "Base URL",
        value=config.DEFAULT_OPENAI_BASE_URL,
        placeholder="https://api.deepseek.com/v1",
        help="大模型 API 端点基础路径"
    )
    model_name = st.text_input(
        "模型名称",
        value=config.DEFAULT_MODEL_NAME,
        placeholder="deepseek-chat",
        help="指定调用的模型标识符"
    )
    temperature = st.slider(
        "生成严谨度 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=config.DEFAULT_TEMPERATURE,
        step=0.05,
        help="值越低回答越聚焦于文档内容、随机性越小"
    )

    st.markdown("---")

    # 2. 向量检索配置
    st.markdown("**2. 向量检索模式**")
    LOCAL_MODE_LABEL = "本地 TF-IDF 字符检索 (默认，无需 Embedding API)"
    embedding_mode = st.radio(
        "Embedding 向量方案",
        options=[LOCAL_MODE_LABEL, "在线 Embedding API (OpenAI 兼容)"],
        index=0,
        help="本地方案基于字符 n-gram 字面相似度，非深度语义向量；载入已保存知识库时需选择与建库时相同的方案"
    )

    use_local = (embedding_mode == LOCAL_MODE_LABEL)
    emb_api_key = api_key
    emb_base_url = base_url
    emb_model = config.DEFAULT_EMBEDDING_MODEL_NAME

    if not use_local:
        emb_api_key = st.text_input("Embedding Key", value=config.DEFAULT_EMBEDDING_API_KEY or api_key, type="password")
        emb_base_url = st.text_input("Embedding URL", value=config.DEFAULT_EMBEDDING_BASE_URL or "https://api.openai.com/v1")
        emb_model = st.text_input("Embedding 模型名", value=config.DEFAULT_EMBEDDING_MODEL_NAME)

    st.markdown("---")

    # 3. 知识库切片参数
    st.markdown("**3. 语义切片与检索参数**")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chunk_size = st.number_input("切片大小 (字符)", min_value=200, max_value=2000, value=config.DEFAULT_CHUNK_SIZE, step=50)
    with col_c2:
        chunk_overlap = st.number_input("重叠跨度 (字符)", min_value=0, max_value=500, value=config.DEFAULT_CHUNK_OVERLAP, step=20)
    
    top_k = st.slider("检索召回数量 (Top-K)", min_value=1, max_value=8, value=config.DEFAULT_TOP_K)

    st.markdown("---")

    # 4. 知识库运行状态
    st.markdown("**4. 知识库存储状态**")
    if st.session_state.vector_store is not None:
        st.markdown(f'<span class="status-pill status-pill-success">● 知识库活跃中</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-pill">○ 等待上传文档</span>', unsafe_allow_html=True)
        
    st.markdown(f"- 索引文档数：**{len(st.session_state.processed_files)}** 篇")
    st.markdown(f"- 向量片段数：**{st.session_state.total_chunks}** 个")

    # 本地持久化知识库热加载
    has_saved_db = (config.VECTOR_DB_DIR / "docs.json").exists() and (config.VECTOR_DB_DIR / "vectors.npy").exists()
    if has_saved_db and st.session_state.vector_store is None:
        if st.button("🔄 载入已保存的本地知识库", use_container_width=True):
            try:
                emb_inst = get_embedding_model(
                    api_key=emb_api_key,
                    base_url=emb_base_url,
                    model_name=emb_model,
                    use_local_fallback=use_local
                )
                loaded_store = SmartVectorStore.load_local(config.VECTOR_DB_DIR, embedding_function=emb_inst)
                st.session_state.vector_store = loaded_store
                if api_key:
                    llm_inst = get_chat_model(
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        temperature=temperature,
                        streaming=True
                    )
                    st.session_state.rag_engine = RAGEngine(
                        llm=llm_inst,
                        vector_store=loaded_store,
                        top_k=top_k
                    )
                st.session_state.processed_files = list(set(d.metadata.get("source", "未知") for d in loaded_store.documents))
                st.session_state.total_chunks = len(loaded_store.documents)
                st.success(f"已恢复本地知识库（{len(st.session_state.processed_files)} 篇文档，{st.session_state.total_chunks} 个切片）！")
                st.rerun()
            except Exception as ex:
                st.error(f"载入本地知识库失败: {ex}")

    if st.button("🗑️ 清空知识库与重置对话", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.rag_engine = None
        st.session_state.processed_files = []
        st.session_state.total_chunks = 0
        st.success("知识库与对话已成功重置！")
        st.rerun()


# ==============================================================================
# 主界面 Header
# ==============================================================================
st.markdown("""
<div class="header-container">
    <div class="brand-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
        </svg>
    </div>
    <div>
        <h1 class="brand-title">SmartDoc-RAG 智能文档知识库助手</h1>
        <p class="brand-subtitle">基于 LangChain LCEL 与在线大模型，实现多格式文档解析、检索增强问答与来源溯源</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 文档管理与知识库构建模块
# ==============================================================================
with st.expander("📂 知识库文件管理与向量构建", expanded=(st.session_state.vector_store is None)):
    uploaded_files = st.file_uploader(
        "支持批量上传 PDF、Word (.docx)、Markdown (.md)、TXT 格式文档",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True
    )

    build_btn = st.button("⚡ 解析并构建向量知识库", type="primary")

    if build_btn:
        if not uploaded_files:
            st.warning("请先选择需要上传的文档。")
        elif not api_key:
            st.warning("请在左侧侧边栏填写大模型 API Key。")
        else:
            with st.spinner("正在提取多源文档、进行语义切片与向量索引构建..."):
                saved_paths = []
                for uf in uploaded_files:
                    target_path = config.UPLOAD_DIR / uf.name
                    with open(target_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_paths.append(target_path)

                # 解析与切分文档
                split_docs, parse_report = load_and_split_files(
                    saved_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                if parse_report["failed"]:
                    for item in parse_report["failed"]:
                        st.warning(f"⚠️ 文件「{item['file']}」未能解析成功: {item['reason']}")

                if not split_docs:
                    st.error("未能从所选文档中提取出有效文本，请检查文档内容！")
                else:
                    # 初始化 Embedding 模型并构建、持久化向量库
                    try:
                        emb_model_instance = get_embedding_model(
                            api_key=emb_api_key,
                            base_url=emb_base_url,
                            model_name=emb_model,
                            use_local_fallback=use_local
                        )
                        v_store = SmartVectorStore.from_documents(split_docs, emb_model_instance)
                        v_store.save_local(config.VECTOR_DB_DIR)
                    except Exception as ex:
                        st.error(f"向量索引构建失败（未切换到其他向量化方式，请检查 Embedding 配置后重试）: {mask_secret(str(ex))}")
                        st.stop()

                    # 初始化大模型与 RAG 引擎
                    llm_instance = get_chat_model(
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        temperature=temperature,
                        streaming=True
                    )
                    rag_engine = RAGEngine(
                        llm=llm_instance,
                        vector_store=v_store,
                        top_k=top_k
                    )

                    # 更新全局会话状态
                    success_file_names = [f["file"] for f in parse_report["success"]]
                    st.session_state.vector_store = v_store
                    st.session_state.rag_engine = rag_engine
                    st.session_state.processed_files = success_file_names
                    st.session_state.total_chunks = len(split_docs)

                    st.success(f"✅ 知识库构建成功！成功解析 {len(success_file_names)} 篇文档，切分为 {len(split_docs)} 个文本片段。")


# ==============================================================================
# 智能对话与实时检索问答模块
# ==============================================================================
st.markdown("### 💬 智能文档问答")

# 快捷问答建议
st.markdown("<div style='font-size: 0.88rem; color: #64748B; margin-bottom: 8px;'>快速提问启发：</div>", unsafe_allow_html=True)
col_q1, col_q2, col_q3 = st.columns(3)
preset_query = None
with col_q1:
    if st.button("📌 总结文档的核心主旨与关键内容", use_container_width=True):
        preset_query = "请全面总结所提供文档的核心主题、主要脉络与关键观点。"
with col_q2:
    if st.button("🔍 梳理文档涉及的关键技术或要点", use_container_width=True):
        preset_query = "请详细列出文档中涉及的关键技术要点、核心概念及其具体含义。"
with col_q3:
    if st.button("📝 提炼结论与实施建议", use_container_width=True):
        preset_query = "基于文档内容，提炼出其总结性结论与后续建议事项。"

# 显示历史对话记录与引用卡片
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📑 命中参考依据 ({len(msg['sources'])} 处来源片段)"):
                for idx, (doc, score) in enumerate(msg["sources"]):
                    src_name = doc.metadata.get("source", "未知来源")
                    page = doc.metadata.get("page", 1)
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-meta">
                            <span>📄 来源 {idx+1}：<code>{src_name}</code> (页码: {page})</span>
                            <span class="score-badge" title="余弦相似度（-1~1），反映切片与提问关键词的相关程度，非答案置信度">检索相关度: {score:.4f}</span>
                        </div>
                        <div class="citation-body">{doc.page_content}</div>
                    </div>
                    """, unsafe_allow_html=True)

# 处理用户提问输入
user_input = st.chat_input("针对已上传文档提问（例如：系统设计的核心原则是什么？）...")
if preset_query:
    user_input = preset_query

if user_input:
    if not api_key:
        st.error("请先在左侧侧边栏配置大模型 API Key。")
    elif not st.session_state.rag_engine:
        st.warning("知识库尚未构建，请先在上方展开面板上传文档并点击「构建向量知识库」。")
    else:
        # 显示用户提问
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 动态同步左侧调节的参数
        current_llm = get_chat_model(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            streaming=True
        )
        st.session_state.rag_engine.llm = current_llm
        st.session_state.rag_engine.top_k = top_k

        # 流式回答（注入有限轮数历史上下文，支持上下文指代追问）
        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("正在检索知识库语义片段并推理生成回答..."):
                try:
                    prior_history = st.session_state.chat_history[:-1]
                    stream_generator, sources = st.session_state.rag_engine.stream_query(
                        user_input,
                        chat_history=prior_history
                    )
                    full_response = ""
                    for chunk in stream_generator:
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)

                    # 渲染精美引用卡片
                    if sources:
                        with st.expander(f"📑 命中参考依据 ({len(sources)} 处来源片段)"):
                            for idx, (doc, score) in enumerate(sources):
                                src_name = doc.metadata.get("source", "未知来源")
                                page = doc.metadata.get("page", 1)
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-meta">
                                        <span>📄 来源 {idx+1}：<code>{src_name}</code> (页码: {page})</span>
                                        <span class="score-badge" title="余弦相似度（-1~1），反映切片与提问关键词的相关程度，非答案置信度">检索相关度: {score:.4f}</span>
                                    </div>
                                    <div class="citation-body">{doc.page_content}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # 记录至对话历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })

                except Exception as e:
                    safe_msg = mask_secret(str(e))
                    err_name = type(e).__name__
                    lower_msg = safe_msg.lower()
                    if "authentication" in err_name.lower() or "401" in lower_msg or "invalid api key" in lower_msg:
                        hint = "API Key 无效或已过期，请在侧边栏检查密钥。"
                    elif "ratelimit" in err_name.lower() or "429" in lower_msg:
                        hint = "请求过于频繁或额度不足（限流），请稍后重试或检查账户余额。"
                    elif "timeout" in err_name.lower() or "timed out" in lower_msg:
                        hint = "请求超时，请检查网络或稍后重试。"
                    elif "connection" in err_name.lower():
                        hint = "无法连接到模型服务，请检查 Base URL 与网络。"
                    else:
                        hint = "模型调用失败。"
                    st.error(f"大模型响应失败：{hint}\n\n详细信息（已脱敏）: {err_name}: {safe_msg}")
                    # 回滚本轮用户提问，避免无回答的问题进入后续多轮上下文
                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                        st.session_state.chat_history.pop()
