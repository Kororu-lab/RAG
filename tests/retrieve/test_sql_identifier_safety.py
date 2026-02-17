import os
import sys

from langchain_core.documents import Document
from psycopg2 import sql as pg_sql

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingest.raptor import raptor_finalize
from src.retrieve import bm25_search, rag_retrieve


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


def test_bm25_build_uses_sql_identifier_for_table_name(monkeypatch):
    class FakeBM25:
        def __init__(self, corpus):
            self.corpus = corpus

    conn = RecordingConn(rows=[(1, "hello world")])
    index = bm25_search.BM25Index(index_path="/tmp/test_bm25_index.pkl")

    monkeypatch.setattr(bm25_search, "BM25Okapi", FakeBM25)
    monkeypatch.setattr(bm25_search.BM25Index, "save", lambda self: None)

    index.build_from_db(conn, "unsafe_table_name;drop")

    query_obj, _ = conn.statements[0]
    assert isinstance(query_obj, pg_sql.Composed)
    assert "Identifier" in repr(query_obj)


def test_rag_viking_filter_uses_sql_identifier_for_table_name():
    retriever = rag_retrieve.RAGRetriever.__new__(rag_retrieve.RAGRetriever)
    retriever.conn = RecordingConn(rows=[(1,)])
    retriever.table_name = "unsafe_table_name;drop"

    class Scope:
        path_patterns = ["doc/%"]

    filtered = retriever._viking_bm25_filter([(1, 0.9)], Scope())

    assert filtered == [(1, 0.9)]
    query_obj, params = retriever.conn.statements[0]
    assert isinstance(query_obj, pg_sql.Composed)
    assert "Identifier" in repr(query_obj)
    assert params is not None


def test_rag_sibling_expand_uses_sql_identifier_for_table_name():
    row_meta = {"source_file": "doc/a.txt", "chunk_id": 1, "level": "0"}
    retriever = rag_retrieve.RAGRetriever.__new__(rag_retrieve.RAGRetriever)
    retriever.conn = RecordingConn(rows=[("child content", row_meta)])
    retriever.table_name = "unsafe_table_name;drop"

    docs = [Document(page_content="title", metadata={"source_file": "doc/a.txt", "chunk_id": 0})]
    expanded = retriever._expand_sibling_chunks(docs)

    assert len(expanded) == 2
    query_obj, _ = retriever.conn.statements[0]
    assert isinstance(query_obj, pg_sql.Composed)
    assert "Identifier" in repr(query_obj)


def test_raptor_upsert_sql_uses_identifier_for_table_name():
    query_obj = raptor_finalize.build_upsert_sql("unsafe_table_name;drop")

    assert isinstance(query_obj, pg_sql.Composed)
    assert "Identifier" in repr(query_obj)
