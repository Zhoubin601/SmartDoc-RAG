"""显式执行的小规模真实 API 验证；不被 pytest 自动采集。

只读取用户提供的凭据，报告不保存凭据。默认最多 8 次 HTTP 请求，SDK 重试关闭。
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

import httpx
from streamlit.testing.v1 import AppTest
from src.models import get_chat_model, mask_secret, LocalTfidfEmbeddings
from src.uploads import ingest_uploads
from src.vector_store import SmartVectorStore
from src.rag_engine import RAGEngine


class ObservedStream(httpx.SyncByteStream):
    def __init__(self, wrapped, record):
        self.wrapped, self.record = wrapped, record

    def __iter__(self):
        pending = b""
        for chunk in self.wrapped:
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.strip() == b"data: [DONE]":
                    self.record["sse_done"] = True
                elif line.startswith(b"data: "):
                    try:
                        obj = json.loads(line[6:])
                        self.record["sse_events"] = self.record.get("sse_events", 0) + 1
                        if obj.get("model"):
                            self.record["response_model"] = obj["model"]
                        if obj.get("usage"):
                            self.record["usage"] = obj["usage"]
                        for choice in obj.get("choices", []):
                            if choice.get("finish_reason"):
                                self.record["finish_reason"] = choice["finish_reason"]
                    except (ValueError, TypeError):
                        pass
            yield chunk

    def close(self):
        self.wrapped.close()


def observe_sse(response, record):
    """在 httpx 解压后的字节层观察 SSE，避免把 gzip 数据当成文本。"""
    decoded = response.iter_bytes()
    observer = ObservedStream(decoded, record)
    response.iter_bytes = lambda chunk_size=None: iter(observer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if args.key_file:
        content = args.key_file.read_text(encoding="utf-8-sig")
        matches = re.findall(r"sk-[A-Za-z0-9_-]+", content)
        if len(matches) != 1:
            raise SystemExit("凭据文件必须包含一个可识别的 Key；其他格式请使用环境变量。")
        key = matches[0]
    if not key:
        raise SystemExit("需要 OPENAI_API_KEY 或 --key-file。")
    # 报告不能被意外覆盖；重跑请使用新路径。
    if args.evidence.exists():
        raise SystemExit("证据文件已存在，请指定新路径。")
    evidence = {
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "config": {"base_url": args.base_url, "model": args.model, "max_tokens": 256,
                   "temperature": 0, "timeout_seconds": 45, "max_retries": 0,
                   "embedding": "LocalTfidfEmbeddings", "top_k": 3, "thinking": "disabled"},
        "http_requests": [], "cases": [],
    }
    active = {}

    def on_request(request):
        if len(evidence["http_requests"]) >= 8:
            raise RuntimeError("已达到本次 8 次 HTTP 请求上限。")
        record = {"method": request.method, "url": str(request.url.copy_with(query=None)), "sse_done": False}
        payload = json.loads(request.content)
        record["stream"] = payload.get("stream", False)
        evidence["http_requests"].append(record)
        active[id(request)] = (time.perf_counter(), record)

    def on_response(response):
        started, record = active.pop(id(response.request))
        record.update(status=response.status_code, headers_seconds=round(time.perf_counter() - started, 4))
        record["content_type"] = response.headers.get("content-type", "")
        if record["stream"]:
            observe_sse(response, record)
        elif "json" in response.headers.get("content-type", ""):
            response.read()
            try:
                payload = response.json()
                if payload.get("usage"):
                    record["usage"] = payload["usage"]
                if payload.get("model"):
                    record["response_model"] = payload["model"]
                choices = payload.get("choices", [])
                if choices:
                    record["finish_reason"] = choices[0].get("finish_reason")
            except ValueError:
                pass

    client = httpx.Client(event_hooks={"request": [on_request], "response": [on_response]}, timeout=45)

    def model_factory(**kwargs):
        kwargs.update(timeout=45, max_retries=0, max_tokens=256,
                      extra_body={"thinking": {"type": "disabled"}}, http_client=client)
        return get_chat_model(**kwargs)

    llm = model_factory(api_key=key, base_url=args.base_url, model_name=args.model, temperature=0, streaming=False)

    def case(name, action, validate):
        start = time.perf_counter()
        record = {"name": name}
        try:
            result = action()
            record.update(result)
            record["result"] = "PASS" if validate(result) else "FAIL"
        except Exception as exc:
            record.update(result="FAIL", error=mask_secret(str(exc), [key]), error_type=type(exc).__name__)
        record["seconds"] = round(time.perf_counter() - start, 4)
        evidence["cases"].append(record)
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name + ": " + record["result"], flush=True)
        return record

    def basic():
        message = llm.invoke("请只用中文回答：你好。")
        return {"answer": message.content, "usage": message.usage_metadata, "finish_reason": message.response_metadata.get("finish_reason")}

    case("中文基础连通性", basic, lambda r: bool(re.search(r"[\u4e00-\u9fff]", r["answer"])) and r["finish_reason"] == "stop")

    def invalid_auth():
        bad = model_factory(api_key="sk-invalid-test-key", base_url=args.base_url, model_name=args.model, streaming=False)
        try:
            bad.invoke("你好")
        except Exception as exc:
            return {"error_type": type(exc).__name__, "status": getattr(exc, "status_code", None)}
        return {"status": "unexpected_success"}

    case("无效凭据错误处理", invalid_auth, lambda r: r["status"] == 401)
    manual = Path(__file__).with_name("live_manual.md")
    history = []
    with tempfile.TemporaryDirectory(prefix="smartdoc-live-") as temporary:
        folder = Path(temporary)
        class Upload:
            name = manual.name
            def getbuffer(self):
                return memoryview(manual.read_bytes())
        docs, parse_report = ingest_uploads([Upload()], folder / "uploads", 120, 10)
        store = SmartVectorStore.from_documents(docs, LocalTfidfEmbeddings())
        store.save_local(folder / "db")
        store = SmartVectorStore.load_local(folder / "db")
        evidence["ingestion"] = {"parse_report": parse_report, "chunks": len(docs), "restored": len(store.documents)}
        engine = RAGEngine(llm, store, top_k=3)

        def rag(question, stream=False):
            if stream:
                engine.llm = model_factory(api_key=key, base_url=args.base_url, model_name=args.model, temperature=0, streaming=True)
                gen, sources = engine.stream_query(question, history)
                chunks = list(gen)
                answer = "".join(chunks)
                engine.llm = llm
            else:
                result = engine.query(question, history)
                answer, sources, chunks = result["answer"], result["source_documents"], []
            history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            return {"question": question, "answer": answer, "chunks": len(chunks),
                    "sources": [{"source": d.metadata["source"], "page": d.metadata.get("page"), "score": s, "text": d.page_content} for d, s in sources]}

        direct = case("技术文档直接事实", lambda: rag("岚桥试验站的文档网关监听端口是多少？"), lambda r: "7219" in r["answer"] and bool(r["sources"]))
        # 后续多轮历史来自真实上一轮响应。
        case("多轮指代与流式输出", lambda: rag("它的请求超时时间是多少秒？", True), lambda r: "37" in r["answer"] and r["chunks"] > 1 and bool(r["sources"]))
        case("同义改写", lambda: rag("岚桥站每天什么时候做归档，旧归档多久清理？"), lambda r: "03:17" in r["answer"] and "17" in r["answer"])
        case("跨段落综合", lambda: rag("请同时给出岚桥试验站网关端口和归档失败错误码及处理方法。"), lambda r: all(word in r["answer"] for word in ["7219", "LQ-417", "权限"]))
        refusal = lambda r: any(word in r["answer"] for word in ["未提及", "未提供", "没有提供", "未涉及", "无法"])
        case("文档外无关问题", lambda: rag("火星上有外星飞船吗？"), refusal)

        # 最后一次真实请求经过 Streamlit 应用，而不是直接调用引擎。
        def ui_missing():
            import config
            with patch.object(config, "VECTOR_DB_DIR", folder / "ui-db"), patch.object(config, "UPLOAD_DIR", folder / "ui-uploads"), patch.object(config, "DEFAULT_OPENAI_API_KEY", key), patch.object(config, "DEFAULT_OPENAI_BASE_URL", args.base_url), patch.object(config, "DEFAULT_MODEL_NAME", args.model), patch("src.models.get_chat_model", model_factory), patch("streamlit.file_uploader", lambda *a, **kw: [Upload()]):
                app = AppTest.from_file(str(CODE_DIR / "app.py"), default_timeout=60).run()
                next(b for b in app.button if b.label == "⚡ 解析并构建向量知识库").click().run()
                app.chat_input[0].set_value("岚桥试验站的采购预算是多少？").run(timeout=60)
                if app.exception or app.error:
                    raise RuntimeError("Streamlit 执行失败：" + "; ".join(e.message for e in app.exception) + "; ".join(e.value for e in app.error))
                result = app.session_state["chat_history"][-1]
                return {"question": "岚桥试验站的采购预算是多少？", "answer": result["content"], "assistant_rendered": any(m.value == result["content"] for m in app.markdown), "sources": [d.metadata["source"] for d, _ in result["sources"]]}
        case("资料不存在信息拒答与真实 Streamlit 流式显示", ui_missing, lambda r: refusal(r) and r["assistant_rendered"])

    evidence["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    evidence["request_count"] = len(evidence["http_requests"])
    args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    client.close()
    return 0 if all(c["result"] == "PASS" for c in evidence["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
