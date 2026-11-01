# -*- coding: utf-8 -*-
import sys
import pytest
from pathlib import Path

# 将 Code 目录加入模块搜索路径，确保直接导入 src.*
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@pytest.fixture(autouse=True)
def block_real_api(monkeypatch):
    """普通测试即使环境有真实 Key，也不能发送 HTTP 请求。"""
    import httpx

    def blocked(*args, **kwargs):
        raise AssertionError("普通单元测试禁止访问真实 API")

    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
