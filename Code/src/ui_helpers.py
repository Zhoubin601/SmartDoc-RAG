"""不依赖 Streamlit 的安全 HTML 渲染辅助函数。"""
from html import escape


def citation_html(doc, score: float, index: int) -> str:
    source = escape(str(doc.metadata.get("source", "未知来源")))
    page = escape(str(doc.metadata.get("page", 1)))
    content = escape(doc.page_content)
    return f"""<div class="citation-card">
        <div class="citation-meta">
            <span>📄 来源 {index + 1}：<code>{source}</code> (页码: {page})</span>
            <span class="score-badge" title="余弦相似度（-1~1），不是答案置信度">检索相关度: {score:.4f}</span>
        </div>
        <div class="citation-body">{content}</div>
    </div>"""
