"""BM25 keyword search with RRF fusion support."""

import os
import pickle
import re
from typing import List, Dict, Tuple, Optional

from psycopg2 import sql
from rank_bm25 import BM25Okapi


MAX_INDEX_FILE_SIZE_BYTES = 512 * 1024 * 1024


class BM25Index:
    """BM25 index for hybrid search with vector embeddings."""
    
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[int] = []
        self.corpus: List[List[str]] = []
        self._loaded = False
    
    def tokenize(self, text: str) -> List[str]:
        """Mixed-lingual tokenizer for Korean + English + IPA."""
        if not text:
            return []
        # Split on whitespace and common punctuation, preserving IPA
        tokens = re.split(r'[\s\.,;:!?\[\]\(\)【】「」\-\n\r\t]+', text)
        # Keep tokens with length > 1, lowercase
        return [t.lower() for t in tokens if len(t) > 1]
    
    def build_from_db(self, conn, table_name: str):
        """Build BM25 index from PostgreSQL original_content."""
        print("  Fetching documents for BM25 indexing...")
        with conn.cursor() as cur:
            query_sql = sql.SQL(
                """
                SELECT id,
                       COALESCE(metadata->>'original_content', content) as text
                FROM {}
                """
            ).format(sql.Identifier(table_name))
            cur.execute(query_sql)
            rows = cur.fetchall()
        
        print(f"  Tokenizing {len(rows)} documents...")
        self.doc_ids = [row[0] for row in rows]
        self.corpus = [self.tokenize(row[1] or "") for row in rows]
        self.bm25 = BM25Okapi(self.corpus)
        self._loaded = True
        
        self.save()
        print(f"  BM25 index built: {len(self.doc_ids)} docs -> {self.index_path}")
    
    def save(self):
        """Persist index to disk."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'doc_ids': self.doc_ids,
                'corpus': self.corpus
            }, f)
    
    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if self._loaded:
            return True
        if not os.path.exists(self.index_path):
            print(f"  BM25 index not found: {self.index_path}")
            return False

        try:
            file_size = os.path.getsize(self.index_path)
        except OSError as e:
            print(f"  BM25 index stat failed: {e}")
            self._invalidate()
            return False

        if file_size <= 0:
            print(f"  BM25 index is empty: {self.index_path}")
            self._invalidate()
            return False
        if file_size > MAX_INDEX_FILE_SIZE_BYTES:
            print(
                f"  BM25 index too large ({file_size} bytes > {MAX_INDEX_FILE_SIZE_BYTES}): "
                f"{self.index_path}"
            )
            self._invalidate()
            return False
        
        print(f"  Loading BM25 index from {self.index_path}...")
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  BM25 index load failed: {e}")
            self._invalidate()
            return False

        if not self._validate_payload(data):
            self._invalidate()
            return False

        self.doc_ids = data["doc_ids"]
        self.corpus = data["corpus"]
        try:
            self.bm25 = BM25Okapi(self.corpus)
        except Exception as e:
            print(f"  BM25 index initialize failed: {e}")
            self._invalidate()
            return False

        self._loaded = True
        print(f"  BM25 index loaded: {len(self.doc_ids)} docs")
        return True

    def _invalidate(self):
        self.doc_ids = []
        self.corpus = []
        self.bm25 = None
        self._loaded = False

    def _validate_payload(self, data) -> bool:
        if not isinstance(data, dict):
            print("  Invalid BM25 index payload: expected dict.")
            return False
        if "doc_ids" not in data or "corpus" not in data:
            print("  Invalid BM25 index payload: missing doc_ids/corpus.")
            return False

        doc_ids = data["doc_ids"]
        corpus = data["corpus"]

        if not isinstance(doc_ids, list) or not isinstance(corpus, list):
            print("  Invalid BM25 index payload: doc_ids/corpus must be lists.")
            return False
        if len(doc_ids) != len(corpus):
            print("  Invalid BM25 index payload: doc_ids/corpus length mismatch.")
            return False
        if any(type(doc_id) is not int for doc_id in doc_ids):
            print("  Invalid BM25 index payload: doc_ids must be ints.")
            return False

        for tokens in corpus:
            if not isinstance(tokens, list):
                print("  Invalid BM25 index payload: corpus entries must be token lists.")
                return False
            if any(not isinstance(token, str) for token in tokens):
                print("  Invalid BM25 index payload: tokens must be strings.")
                return False

        return True
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search using BM25.
        Returns list of (doc_id, bm25_score) sorted by score descending.
        """
        if not self._loaded:
            if not self.load():
                return []
        
        tokens = self.tokenize(query)
        if not tokens:
            return []
            
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k by score
        scored = [(self.doc_ids[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(
    vector_results: List[Tuple[int, float]],
    bm25_results: List[Tuple[int, float]],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> Dict[int, float]:
    """
    Merge rankings using Reciprocal Rank Fusion.
    Returns dict of doc_id -> RRF score.
    """
    rrf_scores = {}
    
    # Vector rankings
    for rank, (doc_id, _) in enumerate(vector_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
    
    # BM25 rankings
    for rank, (doc_id, _) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
    
    return rrf_scores
