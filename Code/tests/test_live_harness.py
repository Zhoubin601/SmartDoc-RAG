"""只验证真实联调脚本的观测逻辑，不进行真实 HTTP 调用。"""
import gzip
import importlib.util
from pathlib import Path

import httpx


def test_sse_observer_handles_http_compression():
    path = Path(__file__).parent / "benchmark" / "run_live_api.py"
    spec = importlib.util.spec_from_file_location("live_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":7}}\n\ndata: [DONE]\n\n'
    response = httpx.Response(200, headers={"content-encoding": "gzip", "content-type": "text/event-stream"}, stream=httpx.ByteStream(gzip.compress(body)))
    record = {}
    module.observe_sse(response, record)
    assert list(response.iter_lines())[-2] == "data: [DONE]"
    assert record["sse_events"] == 2 and record["sse_done"] is True
    assert record["usage"]["total_tokens"] == 7 and record["finish_reason"] == "stop"
