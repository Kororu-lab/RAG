import os
import sys

from langchain_core.documents import Document
from psycopg2 import sql as pg_sql

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieve import rag_retrieve


class RecordingCursor:
    def __init__(self, statements, rows=None):
        self._statements = statements
        self._rows = rows or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self._statements.append((query, params))

    def fetchall(self):
        return list(self._rows)


class RecordingConn:
    def __init__(self, rows=None):
        self.statements = []
        self._rows = rows or []

    def cursor(self):
        return RecordingCursor(self.statements, self._rows)


def test_sibling_expansion_uses_configured_table_name():
    retriever = rag_retrieve.RAGRetriever.__new__(rag_retrieve.RAGRetriever)
    retriever.conn = RecordingConn(rows=[])
    retriever.table_name = "custom_table"

    docs = [Document(page_content="title", metadata={"source_file": "doc/a.txt", "chunk_id": 0})]
    expanded = retriever._expand_sibling_chunks(docs)

    assert len(expanded) == 1
    assert retriever.conn.statements
    query_obj, params = retriever.conn.statements[0]
    assert isinstance(query_obj, pg_sql.Composed)
    assert "Identifier" in repr(query_obj)
    assert params is not None


def test_extract_query_metadata_fallback_when_llm_unavailable(monkeypatch, tmp_path):
    def raise_llm(*_args, **_kwargs):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(rag_retrieve, "get_llm", raise_llm)
    monkeypatch.setattr(rag_retrieve.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rag_retrieve.src.utils, "clear_torch_cache", lambda: None)
    monkeypatch.setattr(rag_retrieve.LLMUtility, "unload_model", lambda *_args, **_kwargs: None)

    metadata = rag_retrieve.extract_query_metadata(
        "Explain grammar.",
        {"project": {"data_path": str(tmp_path)}},
    )

    assert metadata["lang"] is None
    assert metadata["topic"] == "General"


def test_parse_detected_languages_none_null_empty():
    known_languages = ["Korean", "English"]

    assert rag_retrieve.parse_detected_languages("", known_languages) is None
    assert rag_retrieve.parse_detected_languages("none", known_languages) is None
    assert rag_retrieve.parse_detected_languages(" null ", known_languages) is None


def test_parse_detected_languages_comma_separated_and_deduped():
    known_languages = ["Korean", "English", "Japanese"]

    parsed = rag_retrieve.parse_detected_languages(
        "Korean, english, Korean, Japanese",
        known_languages,
    )

    assert parsed == ["Korean", "English", "Japanese"]
