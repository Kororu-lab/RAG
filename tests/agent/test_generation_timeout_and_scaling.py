import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent import nodes


def test_generate_with_timeout_passes_request_timeout_to_generate_answer(monkeypatch):
    captured = {}

    def fake_generate_answer(query, context, request_timeout_sec=None):
        captured["query"] = query
        captured["context"] = context
        captured["request_timeout_sec"] = request_timeout_sec
        return "ok"

    monkeypatch.setattr(nodes, "generate_answer", fake_generate_answer)

    result = nodes._generate_with_timeout("q", "ctx", timeout_sec=9)

    assert result == "ok"
    assert captured["query"] == "q"
    assert captured["context"] == "ctx"
    assert captured["request_timeout_sec"] == 9


def test_generate_with_timeout_timeout_exception_returns_fallback(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise RuntimeError("Provider TIMEOUT while waiting for response")

    monkeypatch.setattr(nodes, "generate_answer", raise_timeout)

    result = nodes._generate_with_timeout("q", "ctx", timeout_sec=7)

    assert "정보가 부족합니다" in result
    assert "7초" in result


def test_generate_with_timeout_non_timeout_exception_reraises(monkeypatch):
    def raise_other(*_args, **_kwargs):
        raise RuntimeError("upstream disconnected")

    monkeypatch.setattr(nodes, "generate_answer", raise_other)

    with pytest.raises(RuntimeError, match="upstream disconnected"):
        nodes._generate_with_timeout("q", "ctx", timeout_sec=7)


def test_retrieve_node_dynamic_scaling_for_hyeonsang_vs_aspect_terms(monkeypatch):
    calls = []

    class DummyRetriever:
        def __init__(self):
            self.config = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def retrieve_documents(self, query, k, top_n, metadata):
            calls.append({"query": query, "k": k, "top_n": top_n, "metadata": metadata})
            return []

    monkeypatch.setattr(nodes, "RAGRetriever", DummyRetriever)
    monkeypatch.setattr(nodes, "extract_query_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        nodes.src.utils.LLMUtility, "unload_model", lambda *_args, **_kwargs: None
    )

    def run(query):
        state = {"question": query, "search_count": 0, "documents": []}
        nodes.retrieve_node(state)
        return calls[-1]["k"], calls[-1]["top_n"]

    assert run("이 현상은 무엇인가요?") == (15, 10)
    assert run("시상 체계를 설명해줘") == (30, 15)
    assert run("Explain aspect marking") == (30, 15)
