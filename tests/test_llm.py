"""Ollama backend tests against a fake HTTP layer (no Ollama needed)."""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm  # noqa: E402


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(responses, calls):
    def _open(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append((url, None if isinstance(req, str) else json.loads(req.data)))
        return FakeResponse(responses[url.rsplit("/", 1)[1]])
    return _open


def test_list_and_pick_models(monkeypatch):
    tags = json.dumps({"models": [{"name": "nomic-embed-text:latest"}, {"name": "llama3.2:3b"},
                                  {"name": "qwen2.5:7b-instruct"}]}).encode()
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen({"tags": tags}, []))
    models = llm.OllamaBackend.list_models()
    assert models == ["llama3.2:3b", "qwen2.5:7b-instruct"]          # embedding model skipped
    assert llm.OllamaBackend.best_model(models) == "qwen2.5:7b-instruct"


def test_no_ollama_means_no_backend(monkeypatch):
    def refuse(*a, **k):
        raise llm.urllib.error.URLError("connection refused")
    monkeypatch.setattr(llm.urllib.request, "urlopen", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert llm.pick_backend("auto") is None
    assert llm.pick_backend("none") is None


def test_stream_hides_thinking_and_sets_options(monkeypatch):
    lines = [{"message": {"content": "<think>hmm"}}, {"message": {"content": " still</think>Hello"}},
             {"message": {"content": " there [1]"}}, {"message": {"content": ""}, "done": True}]
    body = "\n".join(json.dumps(x) for x in lines).encode()
    calls = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen({"chat": body}, calls))
    b = llm.OllamaBackend("qwen3:8b")
    assert "".join(b.stream("sys", [{"role": "user", "content": "hi"}])) == "Hello there [1]"
    sent = calls[0][1]
    assert sent["think"] is False and sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["options"]["num_ctx"] >= 8192


def test_complete_strips_think(monkeypatch):
    body = json.dumps({"message": {"content": "<think>x</think> Answer"}}).encode()
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen({"chat": body}, []))
    assert llm.OllamaBackend("llama3.2:3b").complete("s", []) == "Answer"
