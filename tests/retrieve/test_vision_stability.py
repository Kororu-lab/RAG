import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import utils
from src.retrieve import rag_retrieve


def test_resolve_vision_device_and_dtype_mps_fallback_to_cpu(monkeypatch):
    monkeypatch.setattr(utils, "resolve_torch_device", lambda *_args, **_kwargs: "mps")

    device, dtype = utils.resolve_vision_device_and_dtype(
        {
            "embedding": {"device": "auto"},
            "vision": {"mps_fallback_to_cpu": True},
        }
    )

    assert device == "cpu"
    assert dtype == torch.float32


def test_resolve_vision_store_dtype_mapping_and_default():
    assert (
        utils.resolve_vision_store_dtype({"vision": {"ingest_store_dtype": "float32"}})
        == torch.float32
    )
    assert (
        utils.resolve_vision_store_dtype({"vision": {"ingest_store_dtype": "float16"}})
        == torch.float16
    )
    assert (
        utils.resolve_vision_store_dtype({"vision": {"ingest_store_dtype": "bfloat16"}})
        == torch.bfloat16
    )
    assert (
        utils.resolve_vision_store_dtype({"vision": {"ingest_store_dtype": "invalid"}})
        == torch.float32
    )


def test_retrieve_vision_exception_returns_empty_and_cleans_up(monkeypatch):
    class FailingRetriever:
        def initialize(self):
            return None

        def search(self, *_args, **_kwargs):
            raise RuntimeError("vision failure")

    cleanup_calls = {"count": 0}

    monkeypatch.setattr(rag_retrieve, "ColPaliRetriever", FailingRetriever)

    retriever = rag_retrieve.RAGRetriever.__new__(rag_retrieve.RAGRetriever)
    retriever._clean_memory = lambda: cleanup_calls.__setitem__(
        "count", cleanup_calls["count"] + 1
    )

    results = retriever.retrieve_vision("test query", {"lang": "Korean"})

    assert results == []
    assert cleanup_calls["count"] == 1
