from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.uploads import ingest_uploads
from test_security import upload


@pytest.fixture
def app(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "VECTOR_DB_DIR", tmp_path / "db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "DEFAULT_OPENAI_API_KEY", "dummy-test-key")
    monkeypatch.setattr("src.models.get_chat_model", lambda **kwargs: FakeListChatModel(responses=["数据库端口为 7219。 "]))
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: [upload("<说明>.txt", "数据库端口为 7219。".encode())])
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()


def click(app, label):
    return next(b for b in app.button if b.label == label).click().run()


def test_app_build_stream_history_and_actual_mode(app):
    assert not app.exception
    click(app, "⚡ 解析并构建向量知识库")
    assert not app.exception and app.session_state["total_chunks"] == 1
    assert any("当前知识库实际使用：本地 TF-IDF" in i.value for i in app.info)
    app.chat_input[0].set_value("数据库端口是多少？").run()
    assert not app.exception
    assert app.session_state["chat_history"][-1]["content"].strip() == "数据库端口为 7219。"
    assert any("&lt;说明&gt;.txt" in m.value for m in app.markdown)
    app.radio[0].set_value("在线 Embedding API (OpenAI 兼容)").run()
    assert any("当前知识库实际使用：本地 TF-IDF" in i.value for i in app.info)
    assert any("所选模式与当前知识库不同" in w.value for w in app.warning)


def test_app_failed_rebuild_retains_store(app):
    click(app, "⚡ 解析并构建向量知识库")
    old = app.session_state["vector_store"]
    app.radio[0].set_value("在线 Embedding API (OpenAI 兼容)").run()
    next(i for i in app.text_input if i.label == "Embedding URL").set_value("https://api.deepseek.com/v1").run()
    click(app, "⚡ 解析并构建向量知识库")
    assert not app.exception
    assert app.session_state["vector_store"] is old
    assert any("未提供独立 Embedding" in e.value for e in app.error)


def test_app_reset_restore_and_question_without_rebuild(app):
    click(app, "⚡ 解析并构建向量知识库")
    click(app, "🗑️ 清空知识库与重置对话")
    click(app, "🔄 载入已保存的本地知识库")
    assert not app.exception
    app.chat_input[0].set_value("数据库端口是多少？").run()
    assert not app.exception and len(app.session_state["chat_history"]) == 2


def test_model_configuration_error_is_masked_and_history_rolled_back(app, monkeypatch):
    click(app, "⚡ 解析并构建向量知识库")
    def fail(**kwargs):
        raise ValueError("invalid configuration using dummy-test-key")
    monkeypatch.setattr("src.models.get_chat_model", fail)
    app.chat_input[0].set_value("数据库端口？").run()
    assert not app.exception
    assert app.session_state["chat_history"] == []
    assert app.error and "dummy-test-key" not in app.error[0].value


def test_midstream_error_is_masked_and_history_rolled_back(app, monkeypatch):
    click(app, "⚡ 解析并构建向量知识库")
    monkeypatch.setattr("src.models.get_chat_model", lambda **kwargs: FakeListChatModel(responses=["partial text"], error_on_chunk_number=2))
    app.chat_input[0].set_value("数据库端口？").run()
    assert not app.exception
    assert app.error and app.session_state["chat_history"] == []
