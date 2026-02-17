import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieve import bm25_search, colpali_search


def test_bm25_load_rejects_oversized_file(monkeypatch, tmp_path):
    index_path = tmp_path / "bm25.pkl"
    index_path.write_bytes(b"abc")

    monkeypatch.setattr(bm25_search, "MAX_INDEX_FILE_SIZE_BYTES", 1)

    index = bm25_search.BM25Index(index_path=str(index_path))
    assert index.load() is False
    assert index._loaded is False


def test_bm25_load_rejects_invalid_payload_shape(tmp_path):
    index_path = tmp_path / "bm25_invalid.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"unexpected": "payload"}, f)

    index = bm25_search.BM25Index(index_path=str(index_path))
    assert index.load() is False


def test_bm25_load_rejects_invalid_doc_id_types(tmp_path):
    index_path = tmp_path / "bm25_invalid_types.pkl"
    payload = {
        "doc_ids": [1, "2"],
        "corpus": [["good"], ["tokens"]],
    }
    with open(index_path, "wb") as f:
        pickle.dump(payload, f)

    index = bm25_search.BM25Index(index_path=str(index_path))
    assert index.load() is False


def test_safe_torch_load_falls_back_when_weights_only_unsupported(monkeypatch):
    calls = []

    def fake_load(path, map_location=None, **kwargs):
        calls.append({"path": path, "map_location": map_location, **kwargs})
        if kwargs.get("weights_only"):
            raise TypeError("weights_only unsupported")
        return {"loaded": True}

    monkeypatch.setattr(colpali_search.torch, "load", fake_load)

    loaded = colpali_search._safe_torch_load("/tmp/embeddings.pt")

    assert loaded == {"loaded": True}
    assert len(calls) == 2
    assert calls[0]["weights_only"] is True
    assert "weights_only" not in calls[1]


def test_safe_torch_load_prefers_weights_only_when_supported(monkeypatch):
    calls = []

    def fake_load(path, map_location=None, **kwargs):
        calls.append({"path": path, "map_location": map_location, **kwargs})
        return {"loaded": "weights_only"}

    monkeypatch.setattr(colpali_search.torch, "load", fake_load)

    loaded = colpali_search._safe_torch_load("/tmp/embeddings.pt")

    assert loaded == {"loaded": "weights_only"}
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
