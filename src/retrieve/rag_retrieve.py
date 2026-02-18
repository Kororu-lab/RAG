
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import json
import argparse
import psycopg2
from psycopg2 import sql
import time
import re
from typing import List, Dict, Any
from copy import deepcopy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import difflib

from src.llm.factory import get_llm
import src.utils  # Module import for dynamic patching support
from src.utils import LLMUtility

try:
    from src.retrieve.colpali_search import ColPaliRetriever
except ImportError as e:
    print(f"Error importing ColPaliRetriever: {e}")
    ColPaliRetriever = None

try:
    from src.retrieve.bm25_search import BM25Index, reciprocal_rank_fusion
    BM25_AVAILABLE = True
except ImportError as e:
    print(f"BM25 not available: {e}")
    BM25_AVAILABLE = False



def detect_topic_with_llm(query: str, llm) -> str:
    """Detect linguistic domain with a safe default."""
    valid_topics = ["Phonology", "Grammar", "Lexicon", "General"]
    if llm is None:
        return "General"

    topic_prompt = ChatPromptTemplate.from_template(
        """Analyze the user's query and classify it into one of the following Linguistic Domains:
- Phonology (Sounds, Tones, IPA, Phonemes)
- Grammar (Syntax, Morphology, Sentence Structure)
- Lexicon (Words, Vocabulary, Dictionary)
- General (History, Demographics, Metadata, Others)

Query: {query}
Domain (Return ONLY the category name):"""
    )

    try:
        chain_topic = topic_prompt | llm | StrOutputParser()
        topic_result = (chain_topic.invoke({"query": query}) or "").strip()
    except Exception as e:
        print(f"Topic detection failed: {e}")
        return "General"

    topic_result_lower = topic_result.lower()
    for topic in valid_topics:
        if topic.lower() in topic_result_lower:
            return topic

    return "General"


def parse_detected_languages(result: str, known_languages: List[str]) -> Any:
    """Parse LLM language output into None/str/list with exact+fuzzy matching."""
    known_languages_lookup = {lang.lower(): lang for lang in known_languages}
    normalized = (result or "").strip()
    if normalized.lower() in {"", "none", "null"}:
        return None

    def _map_candidate(candidate: str):
        candidate = (candidate or "").strip()
        if not candidate:
            return None

        if candidate in known_languages:
            return candidate

        exact_match = known_languages_lookup.get(candidate.lower())
        if exact_match:
            return exact_match

        matches = difflib.get_close_matches(
            candidate.lower(),
            list(known_languages_lookup.keys()),
            n=1,
            cutoff=0.6,
        )
        if matches:
            mapped = known_languages_lookup[matches[0]]
            print(f"Mapped '{candidate}' to known language '{mapped}'")
            return mapped

        return None

    split_pattern = r"\s*(?:,|/|;|&|\band\b|및|와|과)\s*"
    candidates = [
        candidate.strip()
        for candidate in re.split(split_pattern, normalized, flags=re.IGNORECASE)
        if candidate.strip()
    ]
    if not candidates:
        return None

    detected_langs = []
    for candidate in candidates:
        mapped = _map_candidate(candidate)
        if mapped:
            detected_langs.append(mapped)

    if len(candidates) == 1:
        raw_lower = normalized.lower()
        embedded_hits = []
        for lang_lower, canonical in known_languages_lookup.items():
            pos = raw_lower.find(lang_lower)
            if pos >= 0:
                embedded_hits.append((pos, canonical))
        if len(embedded_hits) >= 2:
            embedded_hits.sort(key=lambda item: item[0])
            detected_langs = [lang for _, lang in embedded_hits]

    detected_langs = list(dict.fromkeys(detected_langs))
    if len(detected_langs) == 1:
        return detected_langs[0]
    if 2 <= len(detected_langs) <= 3:
        return detected_langs
    if len(detected_langs) > 3:
        print(f"Multi-language query: {len(detected_langs)} languages, disabling filter")
    return None


def extract_query_metadata(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts metadata (Language, Topic) from the user query.
    1. Language Detection (Fast Path or LLM)
    2. Topic Fallback (if Language is None)
    
    Returns: {'lang': str|None, 'topic': str|None}
    """
    metadata = {'lang': None, 'topic': None}

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Language Detection
    # Fast Path: Check against known directory names
    data_path = config["project"]["data_path"]
    doc_dir = os.path.join(data_path, "doc")
    known_languages = []
    if os.path.exists(doc_dir):
        known_languages = [d for d in os.listdir(doc_dir) if os.path.isdir(os.path.join(doc_dir, d))]

    # [COMMENTED OUT] Fast Path - kept for future optimization if LLM latency becomes an issue
    # detected_langs = []
    # for lang in known_languages:
    #     if lang in query.lower():
    #         detected_langs.append(lang)
    # if detected_langs:
    #     metadata['lang'] = detected_langs if len(detected_langs) > 1 else detected_langs[0]

    # LLM-based language detection (handles Korean, Chinese, English names)
    llm = None
    print("Using LLM for language detection...")

    try:
        llm = get_llm("retrieval")

        lang_prompt = ChatPromptTemplate.from_template(
            """Analyze the user's query and identify if a specific language is being asked about.
If found, translate it to its standard English Academic name.
If no specific language is targeted, return "None".

- Query: "Language A의 성조는?" -> "Language A"
- Query: "What is the tone in Language B?" -> "Language B"
- Query: "Explain the grammar." -> "None"

Query: {query}
Language Name (English only, no punctuation):"""
        )

        chain = lang_prompt | llm | StrOutputParser()
        result = (chain.invoke({"query": query}) or "").strip()
        print(f"LLM Detected Language: {result}")
        parsed_lang = parse_detected_languages(result, known_languages)
        if isinstance(parsed_lang, list):
            print(f"Multi-language query detected: {parsed_lang}")
        metadata['lang'] = parsed_lang

    except Exception as e:
        print(f"Metadata extraction failed: {e}")

    if not metadata['lang']:
        print("Language not detected. Triggering Topic Fallback...")
        metadata['topic'] = detect_topic_with_llm(query, llm)
        print(f"LLM Detected Topic: {metadata['topic']}")

    # Cleanup
    print("Ensuring VRAM is clean (Unloading Ollama)...")
    if llm:
        del llm
    try:
        LLMUtility.unload_model("retrieval")
    except Exception:
        pass

    src.utils.clear_torch_cache()
    import gc
    gc.collect()
    time.sleep(1)

    return metadata


class RAGRetriever:
    def __init__(self):
        self.config = src.utils.load_config()  # Use module-level for patching support
        self.db_cfg = self.config.get("database", {})
        
        self.embedding_model_name = self.config["embedding"]["model_name"]
        configured_device = self.config.get("embedding", {}).get("device", "auto")
        self.device = src.utils.resolve_torch_device(configured_device)
        if (configured_device or "auto").lower() != self.device:
            print(f"Device fallback: embedding.device='{configured_device}' -> '{self.device}'")
        
        self.reranker_model_name = self.config["reranker"]["model_name"]
        self.use_reranker = self.config["reranker"]["enabled"]
        self.top_n = self.config["reranker"]["top_n"]
        
        # Connect to DB (Lightweight, keep open)
        print("Connecting to Vector Database (PostgreSQL)...")
        self.conn = psycopg2.connect(
            host=self.db_cfg.get('host', 'localhost'),
            port=self.db_cfg.get('port', 5432),
            user=self.db_cfg.get('user', 'user'),
            password=self.db_cfg.get('password', 'password'),
            dbname=self.db_cfg.get('dbname', 'ltdb_rag')
        )
        self.table_name = self.db_cfg.get('table_name', 'linguistics_raptor')
        
        # Hybrid search config
        hybrid_cfg = self.config.get("retrieval", {}).get("hybrid_search", {})
        self.hybrid_enabled = hybrid_cfg.get("enabled", False) and BM25_AVAILABLE
        self.vector_weight = hybrid_cfg.get("vector_weight", 0.6)
        self.bm25_weight = hybrid_cfg.get("bm25_weight", 0.4)
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)
        self.bm25_index_path = hybrid_cfg.get("index_path", "data/bm25_index.pkl")
        
        if self.hybrid_enabled:
            print(f"Hybrid Search: enabled (vector={self.vector_weight}, bm25={self.bm25_weight})")
        
        # Vision and recursive retrieval config
        retrieval_cfg = self.config.get("retrieval", {})
        self.vision_enabled = retrieval_cfg.get("vision_search", True)
        self.recursive_enabled = retrieval_cfg.get("recursive_retrieval", True)

        # Viking routing config
        viking_cfg = retrieval_cfg.get("viking", {})
        self.viking_enabled = viking_cfg.get("enabled", False)
        self.viking_mode = viking_cfg.get("mode", "soft")
        self.viking_max_expansions = viking_cfg.get("max_expansions", 2)
        self.viking_min_hits = viking_cfg.get("min_hits", 3)
        
        self.vision_keywords = ["도표", "table", "chart", "structure", "구조", "IPA", "paradigm", "gloss", "마커", "marker", "예시", "box", "박스"]

    def close(self):
        """Safely close DB connection."""
        conn = getattr(self, "conn", None)
        if conn is None:
            return
        try:
            if not conn.closed:
                conn.close()
        except Exception as e:
            print(f"Error closing database connection: {e}")
        finally:
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _clean_memory(self):
        """Force garbage collection and torch cache clearing."""
        import gc
        gc.collect()
        src.utils.clear_torch_cache()

    def _is_visual_query(self, query: str) -> bool:
        if not ColPaliRetriever or not self.vision_enabled:
            return False
        return any(k in query.lower() for k in self.vision_keywords)

    @staticmethod
    def _build_viking_sql_filter(viking_scope):
        """Build SQL LIKE clause from Viking scope path patterns."""
        if not viking_scope or not viking_scope.path_patterns:
            return "", []
        like_clauses = ["metadata->>'source_file' LIKE %s"
                        for _ in viking_scope.path_patterns]
        path_filter = f" AND ({' OR '.join(like_clauses)})"
        return path_filter, list(viking_scope.path_patterns)

    def _viking_bm25_filter(self, bm25_results, viking_scope):
        """Filter BM25 results to only include docs matching Viking scope patterns.

        Runs a single SQL query to validate which doc_ids have source_file
        matching the Viking path patterns, then keeps only those.
        """
        if not viking_scope or not viking_scope.path_patterns or not bm25_results:
            return bm25_results

        doc_ids = [doc_id for doc_id, _ in bm25_results]
        id_placeholders = ",".join(["%s"] * len(doc_ids))
        like_clauses = ["metadata->>'source_file' LIKE %s"
                        for _ in viking_scope.path_patterns]
        path_where = " OR ".join(like_clauses)

        query_sql = sql.SQL(
            """
            SELECT id FROM {}
            WHERE id IN ({}) AND ({})
            """
        ).format(
            sql.Identifier(self.table_name),
            sql.SQL(id_placeholders),
            sql.SQL(path_where),
        )
        params = doc_ids + list(viking_scope.path_patterns)

        valid_ids = set()
        with self.conn.cursor() as cur:
            cur.execute(query_sql, tuple(params))
            for row in cur.fetchall():
                valid_ids.add(row[0])

        filtered = [(doc_id, score) for doc_id, score in bm25_results
                     if doc_id in valid_ids]
        if len(filtered) < len(bm25_results):
            print(f"Viking BM25 Filter: {len(bm25_results)} → {len(filtered)}")
        return filtered

    @staticmethod
    def _viking_scope_guard(docs, viking_scope, mode="soft"):
        """Post-fusion scope guard: enforce Viking path constraints on merged docs.

        strict: hard-drop all out-of-scope docs.
        soft: keep out-of-scope only when fewer than 3 in-scope docs remain.
        """
        if not viking_scope or not viking_scope.path_patterns:
            return docs

        import fnmatch

        def _matches_scope(doc):
            src = doc.metadata.get("source_file", "")
            for pattern in viking_scope.path_patterns:
                glob_pat = pattern.replace("%", "*")
                if fnmatch.fnmatch(src, glob_pat):
                    return True
            return False

        in_scope = [d for d in docs if _matches_scope(d)]
        out_scope = [d for d in docs if not _matches_scope(d)]

        if mode == "strict":
            if len(in_scope) < len(docs):
                print(f"Viking Scope Guard (strict): {len(docs)} → {len(in_scope)}")
            return in_scope

        # soft: allow out-of-scope when in-scope count is too low
        if len(in_scope) >= 3:
            if out_scope:
                print(f"Viking Scope Guard (soft): dropped {len(out_scope)} out-of-scope")
            return in_scope

        print(f"Viking Scope Guard (soft): kept {len(out_scope)} out-of-scope "
              f"(only {len(in_scope)} in-scope)")
        return in_scope + out_scope

    def retrieve_text(self, query: str, k: int, metadata: Dict[str, Any] = None,
                      viking_scope=None):
        print(f"Loading Embeddings ({self.embedding_model_name})...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': self.config["embedding"]["normalize_embeddings"]}
        )
        
        query_vector = embeddings.embed_query(query)
        
        del embeddings
        self._clean_memory()
        print("Embeddings unloaded.")

        threshold_dist = 0.55
        fetch_k = k * 3
        
        # Build language filter clause
        lang_filter = ""
        lang_param = []
        if metadata and metadata.get('lang'):
            lang = metadata['lang']
            if isinstance(lang, list):
                # Multi-language query - use IN clause
                placeholders = ','.join(['%s'] * len(lang))
                lang_filter = f" AND metadata->>'lang' IN ({placeholders})"
                lang_param = lang
                print(f"Applying Multi-Language Filter: {lang}")
            else:
                # Single language
                print(f"Applying Language Filter: {lang}")
                lang_filter = " AND metadata->>'lang' = %s"
                lang_param = [lang]
        
        # 1. Vector Search (with optional Viking scope filter)
        viking_filter, viking_params = self._build_viking_sql_filter(viking_scope)
        if viking_filter:
            print(f"Viking Filter: {len(viking_scope.path_patterns)} path patterns")

        min_viking_hits = self.viking_min_hits
        viking_attempts = 0
        fallback_level = 0  # 0=initial, 1=category, 2=lang_only, 3=unrestricted

        while True:
            vector_sql = sql.SQL(
                """
                SELECT id, content, metadata, (embedding <=> %s::vector) as distance
                FROM {}
                WHERE (embedding <=> %s::vector) < %s {} {}
                ORDER BY distance ASC LIMIT %s
                """
            ).format(
                sql.Identifier(self.table_name),
                sql.SQL(lang_filter),
                sql.SQL(viking_filter),
            )
            vector_params = ([query_vector, query_vector, threshold_dist]
                             + lang_param + viking_params + [fetch_k])

            vector_results = []  # [(id, distance)]
            doc_cache = {}       # id -> (content, metadata)
            vector_ranks = {}    # id -> rank (for preserving top vector results)

            with self.conn.cursor() as cur:
                cur.execute(vector_sql, tuple(vector_params))
                for rank, row in enumerate(cur.fetchall()):
                    doc_id, content, meta, dist = row
                    vector_results.append((doc_id, 1 - dist))  # Convert to similarity
                    doc_cache[doc_id] = (content, meta)
                    vector_ranks[doc_id] = rank  # Track original vector rank

            print(f"Vector Search: {len(vector_results)} matches")

            # Viking soft fallback: widen scope if too few results
            if (viking_scope and viking_scope.path_patterns
                    and len(vector_results) < min_viking_hits
                    and self.viking_mode == "soft"
                    and viking_attempts < self.viking_max_expansions):
                from src.retrieve.viking_router import widen_scope
                from src.retrieve.viking_index import get_taxonomy_index
                tax_index = get_taxonomy_index(self.config["project"]["data_path"])
                prev_trace = viking_scope.trace[-1] if viking_scope.trace else "?"
                viking_scope = widen_scope(viking_scope, tax_index)
                viking_filter, viking_params = self._build_viking_sql_filter(viking_scope)
                viking_attempts += 1
                fallback_level += 1
                print(f"Viking Soft Fallback (attempt {viking_attempts}/{self.viking_max_expansions}): "
                      f"trigger=vector_hits({len(vector_results)})<min_hits({min_viking_hits}), "
                      f"level={fallback_level}, "
                      f"step=[{viking_scope.trace[-1] if viking_scope.trace else '?'}], "
                      f"patterns={len(viking_scope.path_patterns)}, "
                      f"min_hits={min_viking_hits}, "
                      f"max_expansions={self.viking_max_expansions}")
                continue

            break  # Enough results or no more fallback
        
        # 2. BM25 Search (if enabled) with Viking scope parity
        bm25_results = []
        if self.hybrid_enabled:
            try:
                bm25_index = BM25Index(self.bm25_index_path)
                bm25_results = bm25_index.search(query, top_k=fetch_k)
                print(f"BM25 Search: {len(bm25_results)} matches")
                # In strict mode, pre-filter BM25 to enforce scope purity.
                # In soft mode, skip pre-filter — let RRF merge freely;
                # post-fusion _viking_scope_guard handles soft enforcement.
                if self.viking_mode == "strict":
                    bm25_results = self._viking_bm25_filter(bm25_results, viking_scope)
            except Exception as e:
                print(f"BM25 search failed: {e}")
        
        # 3. Merge results
        if self.hybrid_enabled and bm25_results:
            # RRF Fusion
            rrf_scores = reciprocal_rank_fusion(
                vector_results, bm25_results,
                k=self.rrf_k,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight
            )
            
            # Sort by RRF score
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            
            # Fetch any docs from BM25 not in cache
            missing_ids = [doc_id for doc_id in sorted_ids if doc_id not in doc_cache]
            if missing_ids:
                placeholders = ','.join(['%s'] * len(missing_ids))
                with self.conn.cursor() as cur:
                    query_sql = sql.SQL(
                        """
                        SELECT id, content, metadata FROM {}
                        WHERE id IN ({})
                        """
                    ).format(
                        sql.Identifier(self.table_name),
                        sql.SQL(placeholders),
                    )
                    cur.execute(query_sql, tuple(missing_ids))
                    for row in cur.fetchall():
                        doc_id, content, meta = row
                        doc_cache[doc_id] = (content, meta)
            
            # Build final doc list
            text_docs = []
            for doc_id in sorted_ids[:fetch_k]:
                if doc_id not in doc_cache:
                    continue
                content, meta = doc_cache[doc_id]
                actual_content = meta.get('original_content', content)
                doc = Document(page_content=actual_content, metadata=meta)
                doc.metadata['score'] = rrf_scores[doc_id]
                doc.metadata['_db_id'] = doc_id
                doc.metadata['_vector_rank'] = vector_ranks.get(doc_id, 999)  # Preserve vector rank
                text_docs.append(doc)
            
            print(f"Hybrid Search (RRF): {len(text_docs)} merged results")
        else:
            # Vector-only fallback
            text_docs = []
            for rank, (doc_id, score) in enumerate(vector_results):
                content, meta = doc_cache[doc_id]
                actual_content = meta.get('original_content', content)
                doc = Document(page_content=actual_content, metadata=meta)
                doc.metadata['score'] = score
                doc.metadata['_db_id'] = doc_id
                doc.metadata['_vector_rank'] = rank  # Preserve vector rank
                text_docs.append(doc)
            
            hybrid_note = ("BM25 filtered to 0 by Viking strict"
                          if self.hybrid_enabled else "hybrid disabled")
            print(f"Vector Search: {len(text_docs)} results ({hybrid_note})")
        
        # Post-fusion Viking scope guard
        text_docs = self._viking_scope_guard(text_docs, viking_scope, self.viking_mode)

        return self._recursive_expand(text_docs)

    def _recursive_expand(self, docs: List[Document]) -> List[Document]:
        """
        [Recursive Retrieval Strategy]
        If a retrieved doc is Level 1 (Summary), expand it to its child Level 0 (Raw) chunks.
        This ensures the LLM sees the original detailed text, not just the summary.
        """
        if not self.recursive_enabled:
            return docs
            
        expanded_docs = []
        l1_found = False
        
        # Collect all L1 child references
        # Set of (source_file, chunk_id)
        child_refs = set()
        
        for doc in docs:
            level = str(doc.metadata.get('level', '0'))
            
            if level == '0':
                expanded_docs.append(doc)
            elif level == '1':
                l1_found = True
                # Parse child_chunks. 
                # Old Schema: [int, int] (Legacy, can't recursive expand reliably without file path)
                # New Schema: [{"source_file": "...", "chunk_id": 0}, ...]
                children = doc.metadata.get('child_chunks', [])
                
                # Check schema by looking at first element
                if children and isinstance(children[0], dict):
                    for child in children:
                        src = child.get("source_file")
                        cid = child.get("chunk_id")
                        if src is not None and cid is not None:
                            child_refs.add((src, int(cid)))
                else:
                    # Legacy schema or empty. Keep L1 node as fallback.
                    # print("Warning: L1 node has legacy schema. Keeping summary.")
                    expanded_docs.append(doc)

        if not child_refs:
            return expanded_docs
            
        print(f"Recursive Retrieval: Expanding {len(child_refs)} L0 chunks from L1 nodes...")
        
        docs_l0 = []
        with self.conn.cursor() as cur:
            from collections import defaultdict
            file_map = defaultdict(list)
            for src, cid in child_refs:
                file_map[src].append(cid)
            
            or_clauses = []
            params = []
            for src, cids in file_map.items():
                placeholders = ",".join(["%s"] * len(cids))
                or_clauses.append(f"(metadata->>'source_file' = %s AND (metadata->>'chunk_id')::int IN ({placeholders}))")
                params.append(src)
                params.extend(cids)
            
            if or_clauses:
                where_clause = " OR ".join(or_clauses)
                query_sql = sql.SQL(
                    "SELECT content, metadata FROM {} WHERE ({})"
                ).format(
                    sql.Identifier(self.table_name),
                    sql.SQL(where_clause),
                )
                
                cur.execute(query_sql, tuple(params))
                rows = cur.fetchall()
                
                for row in rows:
                    content, meta = row
                    # Verify level 0
                    if str(meta.get('level')) == '0':
                         # Use original_content if available, fallback to summary
                         actual_content = meta.get('original_content', content)
                         doc = Document(page_content=actual_content, metadata=meta)
                         # Inherit score? No, this is raw retrieval. 
                         # We will let Reranker handle scoring.
                         docs_l0.append(doc)
        
        # Deduplicate Expanded Docs vs Already Retrieved L0
        # Use (source_file, chunk_id) as key
        existing_keys = set()
        for d in expanded_docs:
            k = (d.metadata.get('source_file'), d.metadata.get('chunk_id'))
            existing_keys.add(k)
            
        final_docs = expanded_docs
        added_count = 0
        
        for d in docs_l0:
            k = (d.metadata.get('source_file'), d.metadata.get('chunk_id'))
            if k not in existing_keys:
                final_docs.append(d)
                existing_keys.add(k)
                added_count += 1
                
        print(f"Recursive Retrieval: Added {added_count} unique L0 chunks. Total Context: {len(final_docs)}")
        return final_docs

    def _expand_sibling_chunks(self, docs: List[Document]) -> List[Document]:
        """
        [Sibling Chunk Expansion]
        When chunk:0 (title) is selected by reranker, also fetch chunks 1-N from same source.
        This ensures detailed content/examples are included, not just titles.
        """
        # Collect source files where chunk:0 was selected
        sources_with_title_only = set()
        existing_chunks = {}  # source_file -> set of chunk_ids
        
        for doc in docs:
            src = doc.metadata.get('source_file', '')
            chunk_id = doc.metadata.get('chunk_id', -1)
            if src not in existing_chunks:
                existing_chunks[src] = set()
            existing_chunks[src].add(chunk_id)
        
        # Find sources where we only have chunk 0
        for src, chunks in existing_chunks.items():
            if chunks == {0} or chunks == {'0'}:
                sources_with_title_only.add(src)
        
        if not sources_with_title_only:
            return docs
            
        print(f"Sibling Expansion: {len(sources_with_title_only)} sources have only title chunk")
        
        # Fetch sibling chunks from database
        sibling_docs = []
        with self.conn.cursor() as cur:
            for src in sources_with_title_only:
                query_sql = sql.SQL(
                    """
                    SELECT content, metadata
                    FROM {}
                    WHERE metadata->>'source_file' = %s
                    AND (metadata->>'chunk_id')::int > 0
                    ORDER BY (metadata->>'chunk_id')::int
                    LIMIT 10
                    """
                ).format(sql.Identifier(self.table_name))
                cur.execute(query_sql, (src,))
                
                for row in cur.fetchall():
                    content, meta = row
                    actual_content = meta.get('original_content', content)
                    doc = Document(page_content=actual_content, metadata=meta)
                    sibling_docs.append(doc)
        
        print(f"Sibling Expansion: Added {len(sibling_docs)} content chunks")
        return docs + sibling_docs

    def retrieve_vision(self, query: str, metadata: Dict[str, Any]):
        vision_retriever = None
        try:
            print("Initializing Vision Retriever (ColPali)...")
            vision_retriever = ColPaliRetriever()
            vision_retriever.initialize()

            # Use lang from metadata as filter
            lang_filter = metadata.get('lang')
            return vision_retriever.search(query, top_k=3, lang_filter=lang_filter)
        except Exception as e:
            print(f"Vision retrieval failed: {e}")
            return []
        finally:
            if vision_retriever is not None:
                try:
                    del vision_retriever
                except Exception:
                    pass
            self._clean_memory()
            print("Vision Retriever unloaded.")

    def perform_rerank(self, query: str, docs: List[Document], top_n: int = None):
        if top_n is None:
            top_n = self.top_n
        
        # Preserve vector top-5 (using _vector_rank metadata)
        vector_top = [d for d in docs if d.metadata.get('_vector_rank', 999) < 5]
            
        print(f"Loading Reranker ({self.reranker_model_name})...")
        reranker = CrossEncoder(
            self.reranker_model_name, 
            device=self.device,
            trust_remote_code=True
        )
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs, batch_size=4) 
        
        del reranker
        self._clean_memory()
        print("Reranker unloaded.")
        
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_n_docs = [doc for doc, score in scored_docs[:top_n]]
        
        # Merge preserved vector top with reranked results
        if vector_top:
            seen_keys = set()
            for d in top_n_docs:
                key = (d.metadata.get('source_file'), d.metadata.get('chunk_id'))
                seen_keys.add(key)
            
            added = 0
            for d in vector_top:
                key = (d.metadata.get('source_file'), d.metadata.get('chunk_id'))
                if key not in seen_keys:
                    top_n_docs.append(d)
                    seen_keys.add(key)
                    added += 1
            if added > 0:
                print(f"Vector Top Preserved: {added} docs")
        
        print(f"Reranked: Reduced from {len(docs)} to {len(top_n_docs)} docs (Top-N={top_n})")
        return top_n_docs

    def retrieve_documents(self, query: str, metadata: Dict[str, Any] = None, k: int = None, top_n: int = None) -> List[Document]:
        """
        Retrieves and returns a list of LangChain Document objects (Text + Vision).
        This method is designed for Agentic use (programmatic access).
        """
        print(f"Retrieving for query: {query}")
        
        # Determine K
        if k is None:
            k = self.config["retrieval"]["k"]
        print(f"Dynamic Top-K: {k}")
        
        # Determine Top-N
        if top_n is None:
            top_n = self.config["reranker"]["top_n"]
        print(f"Dynamic Rerank Top-N: {top_n}")

        if metadata:
            print(f"Metadata Context: {metadata}")
        else:
            metadata = {}

        # Viking routing (search-space restriction)
        viking_scope = None
        if self.viking_enabled:
            from src.retrieve.viking_router import compute_scope
            from src.retrieve.viking_index import get_taxonomy_index
            data_path = self.config["project"]["data_path"]
            tax_index = get_taxonomy_index(data_path)
            viking_scope = compute_scope(query, metadata, tax_index)
            print(f"Viking Scope: conf={viking_scope.confidence}, "
                  f"patterns={len(viking_scope.path_patterns)}, "
                  f"trace={viking_scope.trace}")

        # 1. Text Retrieval (Sequential)
        text_docs = self.retrieve_text(query, k, metadata=metadata,
                                       viking_scope=viking_scope)
        print(f"Text Matches: {len(text_docs)}")

        # Stamp Viking scope metadata on every doc for UI transparency
        if viking_scope:
            scope_meta = {
                "languages": viking_scope.languages,
                "categories": viking_scope.categories,
                "phenomena": viking_scope.phenomena,
                "confidence": viking_scope.confidence,
                "trace": viking_scope.trace,
                "patterns": len(viking_scope.path_patterns),
            }
            for doc in text_docs:
                doc.metadata["_viking_scope"] = scope_meta

        # 2. Vision Retrieval (Sequential)
        vision_docs = []
        if self._is_visual_query(query):
            print(f"Triggering Vision Search (ColPali)... Metadata: {metadata}")
            visual_results = self.retrieve_vision(query, metadata)
            print(f"Vision Search Matches: {len(visual_results)}")
            
            for res in visual_results:
                meta = deepcopy(res.get("metadata") or {})
                # Create a pseudo-Document for Vision
                content = f"[Vision Content] Type: {meta.get('element_type')}\n"
                if 'context_preview' in meta:
                    content += f"Text in Image: {meta['context_preview']}\n"
                
                # Add score to metadata
                meta['score'] = res.get('score')
                meta['type'] = 'vision'
                
                doc = Document(page_content=content, metadata=meta)
                vision_docs.append(doc)

        # 3. Reranking (Sequential)
        if self.use_reranker and text_docs:
            print("Reranking text docs...")
            text_docs = self.perform_rerank(query, text_docs, top_n=top_n)
            # Expand sibling chunks for docs where only chunk:0 (title) was selected
            text_docs = self._expand_sibling_chunks(text_docs)

        # Merge Text and Vision
        final_docs = text_docs + vision_docs
        
        # Robust Reference ID Generation (Mutates metadata in place)
        for doc in final_docs:
             # Skip if already generated (e.g. Vision might have it)
             if 'ref_id' in doc.metadata: continue

             if doc.metadata.get('type') == 'vision':
                 doc.metadata['ref_id'] = f"{os.path.basename(doc.metadata.get('parent_filename','?'))} (Vision)"
                 continue

             ref_id = doc.metadata.get('group_key')
             if not ref_id:
                # Fallback for L0: file_basename:chunk_id (Header)
                src_file = doc.metadata.get('source_file', 'Unknown')
                src_base = os.path.basename(src_file)
                lang = doc.metadata.get('lang')
                
                if lang and lang.lower() not in ['unknown', 'none']:
                    display_name = f"{lang}/{src_base}"
                else:
                    display_name = src_base
                    
                chunk_id = doc.metadata.get('chunk_id', '?')
                header = doc.metadata.get('original_header', '')
                
                ref_id = f"{display_name}:{chunk_id}"
                if header:
                    ref_id += f" [{header}]"
             
             doc.metadata['ref_id'] = str(ref_id)

        return final_docs

    def retrieve(self, query: str, metadata: Dict[str, Any] = None):
        """
        Legacy wrapper for CLI usage. Saves context.json.
        """
        final_docs = self.retrieve_documents(query, metadata)
        
        # Format for JSON output
        text_context_lines = []
        vision_context_lines = []
        references = []
        
        seen = set()
        
        for doc in final_docs:
            # Context String Construction
            if doc.metadata.get('type') == 'vision':
                vision_context_lines.append(doc.page_content)
                
                # Reference
                ref_id = doc.metadata.get('ref_id')
                if ref_id not in seen:
                    references.append({
                        "type": "vision",
                        "element_type": doc.metadata.get('element_type'),
                        "score": doc.metadata.get('score'),
                        "file": doc.metadata.get('parent_filename'),
                        "image": doc.metadata.get('image_path')
                    })
                    seen.add(ref_id)
            else:
                # Text
                level = doc.metadata.get('level', '0')
                prefix = "[Summary]" if str(level) == '1' else "[Detail]"
                text_context_lines.append(f"{prefix} Source: {doc.metadata.get('ref_id', 'Chunk')}\n{doc.page_content}")
                
                # Reference
                ref_id = doc.metadata.get('ref_id')
                if ref_id not in seen:
                    references.append({
                        "type": "text",
                        "level": level,
                        "ref_id": ref_id,
                        "preview": doc.page_content[:50] + "..."
                    })
                    seen.add(ref_id)

        full_context = "\n\n".join(text_context_lines)
        if vision_context_lines:
             full_context += "\n\n[Visual Evidence Found]\n" + "\n---\n".join(vision_context_lines)

        payload = {
            "query": query,
            "metadata": metadata, 
            "context": full_context,
            "references": references
        }
        
        with open("context.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            
        print("Stage 1 Complete. Context saved to context.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User query")
    args = parser.parse_args()
    
    # Pre-Flight: Detect Metadata (Language/Topic)
    config = src.utils.load_config()
    metadata = extract_query_metadata(args.query, config)
    
    # Initialize RAGRetriever (Loads Embeddings, Reranker, ColPali + Postgres)
    with RAGRetriever() as retriever:
        retriever.retrieve(args.query, metadata=metadata)
