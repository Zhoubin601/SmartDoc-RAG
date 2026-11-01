from html.parser import HTMLParser
from io import BytesIO
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.ui_helpers import citation_html
from src.uploads import save_upload, ingest_uploads
from src.models import mask_secret, get_embedding_model, EmbeddingAPIError, SafeOpenAIEmbeddings


def upload(name, content=b"sample text"):
    return SimpleNamespace(name=name, getbuffer=lambda: memoryview(content))


def test_citation_escapes_all_external_fields():
    class Tags(HTMLParser):
        tags = []
        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)
    doc = Document(page_content='<script>alert("X")</script>&正文',
                   metadata={"source": '<img src=x onerror="boom">.txt', "page": "</span><svg onload=x>"})
    rendered = citation_html(doc, .5, 0)
    parser = Tags()
    parser.feed(rendered)
    assert set(parser.tags) <= {"div", "span", "code"}
    assert "&lt;script&gt;" in rendered and "&lt;img" in rendered and "&lt;svg" in rendered
    assert 'class="citation-card"' in rendered and "0.5000" in rendered


@pytest.mark.parametrize("name", ["../escape.txt", r"..\escape.txt", r"C:\evil.txt", "/tmp/evil.md", "bad\x00.txt", "a\n.txt", "C:evil.txt"])
def test_upload_rejects_paths_and_controls(tmp_path, name):
    with pytest.raises(ValueError):
        save_upload(upload(name), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,data", [("a.exe", b"x"), ("a.txt.exe", b"x"), ("empty.txt", b"")])
def test_upload_rejects_invalid_extension_or_empty(tmp_path, name, data):
    with pytest.raises(ValueError):
        save_upload(upload(name, data), tmp_path)


def test_same_name_and_special_characters_preserve_sources(tmp_path):
    name = '中文 <说明> & "示例".TXT'
    docs, report = ingest_uploads([upload(name, "端口为 7219".encode()), upload(name, "备份周期为 17 天".encode())], tmp_path)
    assert len(report["success"]) == len(docs) == 2
    assert {d.metadata["source"] for d in docs} == {name}
    assert len({d.metadata["storage_id"] for d in docs}) == 2
    assert len(list(tmp_path.iterdir())) == 2
    assert {p.read_text(encoding="utf-8") for p in tmp_path.iterdir()} == {"端口为 7219", "备份周期为 17 天"}


def test_upload_collision_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr("src.uploads.uuid4", lambda: SimpleNamespace(hex="same"))
    path = save_upload(upload("x.md", b"original"), tmp_path)
    with pytest.raises(FileExistsError):
        save_upload(upload("x.md", b"changed"), tmp_path)
    assert path.read_bytes() == b"original"


def test_corrupted_upload_does_not_discard_other_file(tmp_path):
    docs, report = ingest_uploads([upload("broken.pdf", b"not PDF"), upload("ok.markdown")], tmp_path)
    assert len(docs) == 1
    assert report["success"][0]["file"] == "ok.markdown"
    assert report["failed"][0]["file"] == "broken.pdf"


@pytest.mark.parametrize("text", ["sk-short", "Bearer arbitrary-secret-123", "api_key='nonstandard123'", '{"api_key": "nonstandard123"}', "token=nonstandard123"])
def test_mask_nonstandard_credentials(text):
    masked = mask_secret(text)
    assert "sk-short" not in masked
    assert "arbitrary-secret-123" not in masked
    assert "nonstandard123" not in masked


def test_explicit_secret_redacts_bare_value():
    assert "opaque-credential" not in mask_secret("failed: opaque-credential", ["opaque-credential"])


def test_embedding_init_failure_no_fallback_or_secret_output(monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("bare opaque-credential")
    monkeypatch.setattr("src.models.OpenAIEmbeddings", fail)
    with pytest.raises(EmbeddingAPIError) as error:
        get_embedding_model(api_key="opaque-credential", base_url="https://example.org/v1")
    assert "opaque-credential" not in str(error.value)
    assert capsys.readouterr().out == ""


def test_online_mode_and_query_error(monkeypatch):
    emb = get_embedding_model(api_key="opaque-credential", base_url="https://example.org/v1", model_name="test-embedding")
    assert isinstance(emb, SafeOpenAIEmbeddings)
    def fail(*args, **kwargs):
        raise RuntimeError("request with opaque-credential failed")
    monkeypatch.setattr(type(emb.raw_embeddings), "embed_query", fail)
    with pytest.raises(EmbeddingAPIError) as error:
        emb.embed_query("hello")
    assert "opaque-credential" not in str(error.value)


@pytest.mark.parametrize("url", ["https://u:p@example.org/v1", "https://example.org/v1?key=secret", "file:///tmp/model"])
def test_embedding_rejects_credential_urls(url):
    with pytest.raises(ValueError):
        get_embedding_model(api_key="dummy", base_url=url)
