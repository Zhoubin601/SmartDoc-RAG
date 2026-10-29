# -*- coding: utf-8 -*-
"""
SmartDoc-RAG 可复现基准评测脚本
读取 sample_manual.md 与 eval_questions.json（共 10 道覆盖 6 类场景的标准测试题），
检验：
1. 文档解析与分块
2. TF-IDF 2~3 gram 向量构建、持久化保存与恢复一致性
3. Top-3 检索命中率（Hit@3）与最高余弦相似度分布
4. 在线大模型（或离线 Mock）端到端生成结果与文档外问题拒答率
"""
import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# 将 Code 目录加入 sys.path
CODE_DIR = Path(__file__).resolve().parent.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.document_loader import load_and_split_files
from src.models import LocalTfidfEmbeddings, get_chat_model
from src.vector_store import SmartVectorStore
from src.rag_engine import RAGEngine


BENCHMARK_DIR = Path(__file__).resolve().parent
MANUAL_PATH = BENCHMARK_DIR / "sample_manual.md"
QUESTIONS_PATH = BENCHMARK_DIR / "eval_questions.json"
REPORT_PATH = BENCHMARK_DIR / "BENCHMARK_REPORT.md"


def run_benchmark():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").strip()
    model_name = os.getenv("MODEL_NAME", "deepseek-chat").strip()

    # 1. 加载并切分手册
    split_docs, parse_report = load_and_split_files([MANUAL_PATH], chunk_size=260, chunk_overlap=40)

    # 2. 构建向量库并测试持久化恢复
    emb = LocalTfidfEmbeddings(max_features=2048)
    raw_store = SmartVectorStore.from_documents(split_docs, emb)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_store.save_local(Path(tmp_dir))
        del raw_store
        store = SmartVectorStore.load_local(Path(tmp_dir))

    # 3. 初始化 LLM
    if api_key:
        llm = get_chat_model(api_key=api_key, base_url=base_url, model_name=model_name, temperature=0.1, streaming=False)
        mode_desc = f"真实在线大模型 API (`{model_name}`)"
    else:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        llm = FakeListChatModel(responses=["[离线 Mock 模式：仅验证检索与提示词组装]"])
        mode_desc = "离线 Mock 模式（未设置 OPENAI_API_KEY，回答列仅为占位，不代表模型效果）"

    engine = RAGEngine(llm=llm, vector_store=store, top_k=3, min_score_threshold=0.08)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    records = []
    in_doc_hit_count = 0
    in_doc_total = 0
    out_doc_refuse_count = 0
    out_doc_total = 0

    for q in questions:
        qid = q["id"]
        qtype = q["type"]
        question = q["question"]
        chat_history = q.get("chat_history", None)
        expected_kws = q["expected_keywords"]
        should_refuse = q["should_refuse"]

        res = engine.query(question, chat_history=chat_history)
        answer = res["answer"].strip().replace("\n", " ")
        sources = res["source_documents"]
        top_score = sources[0][1] if sources else 0.0
        top_snippet = sources[0][0].page_content.strip().replace("\n", " ")[:70] + "..." if sources else "无"
        combined_text = "\n".join(doc.page_content for doc, _ in sources)

        if not should_refuse:
            in_doc_total += 1
            hit = all(kw in combined_text for kw in expected_kws)
            ans_ok = all(kw.lower() in answer.lower() for kw in expected_kws) if api_key else hit
            passed = hit and ans_ok
            if hit:
                in_doc_hit_count += 1
        else:
            out_doc_total += 1
            refused = ("未提及" in answer) or ("没有提及" in answer) or ("不足" in answer) or (top_score < 0.08)
            passed = refused
            if refused:
                out_doc_refuse_count += 1

        records.append({
            "id": qid,
            "type": qtype,
            "question": question,
            "expected": ", ".join(expected_kws) if expected_kws else "拒答（未提及）",
            "top_score": f"{top_score:.4f}",
            "top_snippet": top_snippet,
            "answer": answer[:100] + ("..." if len(answer) > 100 else ""),
            "passed": "PASS" if passed else "FAIL"
        })
        print(f"[{qid}] ({qtype}) TopScore={top_score:.4f} -> {'PASS' if passed else 'FAIL'}")

    # 生成 Markdown 报告
    lines = [
        "# SmartDoc-RAG 基准测试集（10 题）实测评估报告",
        "",
        f"- **评测执行时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **评测模式**：{mode_desc}",
        f"- **测试文档**：`Code/tests/benchmark/sample_manual.md`（切分为 {len(split_docs)} 个文本片段，`chunk_size=260`, `chunk_overlap=40`）",
        f"- **检索方案**：`LocalTfidfEmbeddings`（字符 2~3 gram，`max_features=2048`，余弦相似度 Top-3）",
        f"- **相关性守卫阈值**：最低拒答阈值 `min_score_threshold = 0.08`，低相关度预警阈值 `caution_score_threshold = 0.15`",
        "",
        "## 1. 统计摘要（测试集规模 N = 10）",
        "",
        f"- **文档内问题 Top-3 检索命中率（Hit@3）**：`{in_doc_hit_count} / {in_doc_total}`（在 8 道文档内测试题上达到 {in_doc_hit_count/in_doc_total*100:.1f}%）",
        f"- **文档外无关问题合理拒答率**：`{out_doc_refuse_count} / {out_doc_total}`（在 2 道文档外测试题上达到 {out_doc_refuse_count/out_doc_total*100:.1f}%）",
        "- **局限性说明**：本评测基于 10 道结构化中文测试题与单篇技术手册样本，用于验证系统核心检索链路、指代消解与无资料拒答机制的正确性，不代表在任意大规模复杂语料上的普适准确率。字符级 TF-IDF 对字面词根重合敏感，若遇到完全不含共同词根的极端跨语言或深度抽象改写，需切换至在线稠密语义 Embedding 模式。",
        "",
        "## 2. 逐题评测明细表",
        "",
        "| 题号 | 问题类型 | 测试问题 | 预期答案要点 | Top-1 相似度 | 实际 Top-1 检索片段摘要 | 模型实际回答 | 结果 |",
        "| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: |"
    ]

    for r in records:
        lines.append(
            f"| {r['id']} | {r['type']} | {r['question']} | `{r['expected']}` | `{r['top_score']}` | {r['top_snippet']} | {r['answer']} | **{r['passed']}** |"
        )

    in_scores = [float(r["top_score"]) for r, q in zip(records, questions) if not q["should_refuse"]]
    score_map = {r["id"]: r["top_score"] for r in records}
    lines.extend([
        "",
        "## 3. 阈值选择依据",
        "",
        "根据本次运行的 Q1~Q10 实测 Top-1 余弦相似度分布：",
        f"- **文档内有效问题（Q1~Q8）**：Top-1 相似度分布在 `{min(in_scores):.4f} ~ {max(in_scores):.4f}` 之间；",
        f"- **含局部通用词重合的文档外问题（Q9）**：Top-1 相似度为 `{score_map.get('Q9')}`（落入 `[0.08, 0.15)` 低相关度预警区间时，系统注入预警提示，由模型判断是否拒答）；",
        f"- **完全无关的文档外问题（Q10）**：Top-1 相似度为 `{score_map.get('Q10')}`（低于 `0.08` 时在检索格式化阶段直接触发拒答提示）。",
        "",
        "阈值 `0.08` / `0.15` 依据上述样例分布选取；样例规模较小，更换语料后应重新运行本脚本校准。"
    ])

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n评测报告已生成至: {REPORT_PATH}")


if __name__ == "__main__":
    run_benchmark()
