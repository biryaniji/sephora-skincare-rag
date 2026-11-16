"""LLM backends for answer generation.

  * ClaudeBackend  - Anthropic API (needs ANTHROPIC_API_KEY). Best grounding and refusals.
  * OllamaBackend  - a local model through Ollama (free, runs on your laptop).
                     Install Ollama, then e.g.  ollama pull qwen2.5:7b   (or llama3.2:3b on slower laptops)

pick_backend("auto") uses Claude if a key is set, otherwise the best Ollama model it finds,
otherwise None (the app then lists retrieved products without writing an answer).
Choose explicitly with LLM_BACKEND=claude|ollama|none and OLLAMA_MODEL=<name>.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Iterator

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_FAST_MODEL = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-5-5")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

# Preferred local models, best first. Prefix match, so "qwen2.5:7b" matches "qwen2.5:7b-instruct-q4_K_M".
OLLAMA_PREFERENCE = ["qwen3:8b", "qwen2.5:7b", "llama3.1:8b", "gemma3:12b", "gemma3:4b", "qwen3:4b",
                     "mistral", "llama3.2:3b", "qwen2.5:3b", "phi4-mini", "gemma3", "qwen3", "qwen2.5", "llama3"]
THINKING_MODELS = ("qwen3", "deepseek-r1", "gpt-oss", "magistral")


class LLMBackend:
    name = "none"
    model = ""

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1500, fast: bool = False,
                 json_mode: bool = False) -> str:
        raise NotImplementedError

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1500) -> Iterator[str]:
        yield self.complete(system, messages, max_tokens)

    where = ""

    @property
    def label(self) -> str:
        return f"{self.model} {self.where}".strip()


# ---------------------------------------------------------------- Claude
class ClaudeBackend(LLMBackend):
    name = "Claude"
    where = "via the Anthropic API"

    def __init__(self, model: str = CLAUDE_MODEL, fast_model: str = CLAUDE_FAST_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.fast_model = model, fast_model

    def complete(self, system, messages, max_tokens=1500, fast=False, json_mode=False):
        if fast:
            # Haiku 5.5 has no server-side fallback, so this is a plain call.
            r = self.client.messages.create(model=self.fast_model, max_tokens=max_tokens, system=system,
                                            messages=messages, output_config={"effort": "low"})
        else:
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                output_config={"effort": "low"},          # chat route: fast, short answers
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",                      # if the model declines, retry on a fallback model
            )
        if r.stop_reason == "refusal":
            return "Sorry, I can't help with that request."
        return "".join(b.text for b in r.content if b.type == "text").strip()

    def stream(self, system, messages, max_tokens=1500):
        text = self.complete(system, messages, max_tokens)
        # Reveal in small pieces so the UI feels live.
        for piece in re.findall(r"\S+\s*|\s+", text):
            yield piece


# ---------------------------------------------------------------- Ollama
def _strip_thinking(text: str) -> str:
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()


class OllamaBackend(LLMBackend):
    name = "Ollama"
    where = "on this laptop"

    def __init__(self, model: str, host: str = OLLAMA_HOST, timeout: int = 300):
        self.model, self.host, self.timeout = model, host, timeout

    @staticmethod
    def list_models(host: str = OLLAMA_HOST, timeout: float = 1.5) -> list[str]:
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return []
        return [m["name"] for m in data.get("models", []) if "embed" not in m["name"].lower()]

    @staticmethod
    def best_model(models: list[str]) -> str | None:
        for pref in OLLAMA_PREFERENCE:
            for m in models:
                if m.startswith(pref):
                    return m
        return models[0] if models else None

    def _payload(self, system, messages, max_tokens, stream, json_mode=False):
        body = {"model": self.model, "stream": stream,
                "messages": [{"role": "system", "content": system}] + messages,
                "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": max_tokens}}
        if self.model.startswith(THINKING_MODELS):
            body["think"] = False          # answer directly; much faster on a laptop
        if json_mode:
            body["format"] = "json"        # constrained decoding: always valid JSON
            body["options"]["temperature"] = 0
        return json.dumps(body).encode("utf-8")

    def _post(self, payload: bytes):
        req = urllib.request.Request(f"{self.host}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def complete(self, system, messages, max_tokens=1500, fast=False, json_mode=False):
        with self._post(self._payload(system, messages, max_tokens, stream=False, json_mode=json_mode)) as r:
            data = json.loads(r.read().decode("utf-8"))
        return _strip_thinking(data.get("message", {}).get("content", ""))

    def stream(self, system, messages, max_tokens=1500):
        in_think = False
        with self._post(self._payload(system, messages, max_tokens, stream=True)) as r:
            for line in r:
                if not line.strip():
                    continue
                part = json.loads(line.decode("utf-8"))
                if part.get("error"):
                    raise RuntimeError(part["error"])
                token = part.get("message", {}).get("content", "")
                # Some models still emit <think>...</think>: hide it.
                if "<think>" in token:
                    in_think = True
                if in_think:
                    if "</think>" in token:
                        in_think, token = False, token.split("</think>", 1)[1]
                    else:
                        continue
                if token:
                    yield token
                if part.get("done"):
                    break


# ---------------------------------------------------------------- selection
def pick_backend(preference: str | None = None, ollama_model: str | None = None) -> LLMBackend | None:
    pref = (preference or os.getenv("LLM_BACKEND", "auto")).lower()
    if pref == "none":
        return None
    if pref in ("auto", "claude") and os.getenv("ANTHROPIC_API_KEY"):
        try:
            return ClaudeBackend()
        except ImportError:
            if pref == "claude":
                raise
    if pref in ("auto", "ollama"):
        model = ollama_model or os.getenv("OLLAMA_MODEL")
        if not model:
            model = OllamaBackend.best_model(OllamaBackend.list_models())
        if model:
            return OllamaBackend(model)
    return None
