import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieve import rag_retrieve


def test_vision_metadata_is_copied_without_mutating_source_results():
    source_metadata = {
        "element_type": "table",
        "parent_filename": "grammar.pdf",
        "image_path": "images/table_1.png",
        "nested": {"keep": "original"},
    }
    visual_results = [{"metadata": source_metadata, "score": 0.91}]

    retriever = rag_retrieve.RAGRetriever.__new__(rag_retrieve.RAGRetriever)
    retriever.config = {"retrieval": {"k": 5}, "reranker": {"top_n": 3}}
    retriever.viking_enabled = False
    retriever.use_reranker = False
    retriever._is_visual_query = lambda _query: True
    retriever.retrieve_text = lambda *_args, **_kwargs: []
    retriever.retrieve_vision = lambda *_args, **_kwargs: visual_results

    docs = retriever.retrieve_documents("show me the table", metadata={"lang": "takpa"})

    assert len(docs) == 1
    assert docs[0].metadata["type"] == "vision"
    assert docs[0].metadata["score"] == 0.91
    assert "type" not in source_metadata
    assert "score" not in source_metadata

    docs[0].metadata["nested"]["keep"] = "changed"
    assert source_metadata["nested"]["keep"] == "original"
