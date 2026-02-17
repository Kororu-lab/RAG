import io
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils
from src.eval import run_retrieval_ablation


def test_run_condition_uses_scoped_override_without_leak(monkeypatch):
    seen_hybrid_flags = []

    fake_module = types.ModuleType("src.retrieve.rag_retrieve")

    class FakeRetriever:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def retrieve_documents(self, _query_text, metadata=None):
            cfg = src.utils.load_config()
            seen_hybrid_flags.append(cfg["retrieval"]["hybrid_search"]["enabled"])
            return []

    fake_module.RAGRetriever = FakeRetriever
    monkeypatch.setitem(sys.modules, "src.retrieve.rag_retrieve", fake_module)

    queries = [{"query": "q1", "metadata": {"lang": "korean"}}]
    base_config = {
        "retrieval": {"hybrid_search": {"enabled": False}},
        "reranker": {"enabled": False},
    }
    audit_fp = io.StringIO()

    run_retrieval_ablation.run_condition(
        "cond_true",
        {"retrieval.hybrid_search.enabled": True},
        queries,
        base_config,
        audit_fp,
    )
    run_retrieval_ablation.run_condition(
        "cond_false",
        {"retrieval.hybrid_search.enabled": False},
        queries,
        base_config,
        audit_fp,
    )

    assert seen_hybrid_flags == [True, False]
    assert src.utils._CONFIG_OVERRIDE.get() is None
