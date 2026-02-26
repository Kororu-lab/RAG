
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

from src.llm.factory import get_llm
import src.utils  # Module import for dynamic patching support
from src.utils import LLMUtility
from src.retrieve.language_cards import build_language_cards

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

def parse_detected_languages(result: str, known_languages: List[str], max_languages: int = 3) -> Any:
    """
    Parse LLM detector output strictly as JSON and validate against allowed IDs.
    No alias, pattern, or heuristic matching is used.
    """
    known_languages_lookup = {lang.lower(): lang for lang in known_languages}
    normalized = (result or "").strip()
    if normalized.lower() in {"", "none", "null"}:
        return None

    def _parse_json_payload(text: str):
        candidates = [text]
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            candidates.insert(0, fence_match.group(1).strip())

        bracket_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if bracket_match:
            candidates.append(bracket_match.group(1).strip())

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    structured = _parse_json_payload(normalized)
    candidates: List[str] = []
    if isinstance(structured, dict):
        values = structured.get("languages", [])
        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, list):
            candidates = [str(v).strip() for v in values if str(v).strip()]
    elif isinstance(structured, list):
        candidates = [str(v).strip() for v in structured if str(v).strip()]

    if not candidates:
        return None

    detected_langs: List[str] = []
    for candidate in candidates:
        mapped = known_languages_lookup.get(candidate.lower())
        if mapped:
            detected_langs.append(mapped)

    detected_langs = list(dict.fromkeys(detected_langs))
    if len(detected_langs) == 1:
        return detected_langs[0]
    if 2 <= len(detected_langs) <= max_languages:
        return detected_langs
    if len(detected_langs) > max_languages:
        print(f"Multi-language query: {len(detected_langs)} languages, disabling filter")
    return None


def _as_lang_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _coerce_numeric(value: Any, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= minimum else default
    except Exception:
        return default


def _coerce_float(value: Any, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed >= minimum else default
    except Exception:
        return default


def _invoke_with_retry(chain, payload: Dict[str, Any], retries: int, backoff_sec: float,
                       diagnostics: Dict[str, Any], timeout_location: str) -> str:
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return (chain.invoke(payload) or "").strip()
        except Exception as e:
            print(
                f"Metadata LLM call failed (location={timeout_location}, attempt={attempt}/{attempts}): {e}"
            )
            if _is_timeout_exception(e):
                diagnostics["timeout_location"] = timeout_location
            if attempt >= attempts:
                raise
            sleep_sec = backoff_sec * attempt
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return ""


def _is_timeout_exception(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "timeout" in msg
        or "read timed out" in msg
        or "timed out" in msg
    )


def extract_query_metadata(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts metadata (Language) from the user query.

    Returns: {'lang': str|list|None}
    """
    retrieval_cfg = config.get("retrieval", {})
    diagnostics = {
        "lang_detect_method": "none",
        "detected_languages": [],
        "detected_languages_candidates": [],
        "lang_detect_stage": "none",
        "enable_language_filter": bool(retrieval_cfg.get("enable_language_filter", True)),
        "viking_use_detected_language": bool(retrieval_cfg.get("viking_use_detected_language", True)),
        "viking_language_seeded": False,
        "l0_only": bool(retrieval_cfg.get("l0_only", False)),
        "filter_applied": False,
        "timeout_location": "none",
    }
    metadata = {
        "lang": None,
        "_diagnostics": diagnostics,
    }

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Language candidates come from doc directory names (closed-world set)
    data_path = config["project"]["data_path"]
    doc_dir = os.path.join(data_path, "doc")
    known_languages = []
    if os.path.exists(doc_dir):
        known_languages = [d for d in os.listdir(doc_dir) if os.path.isdir(os.path.join(doc_dir, d))]

    # LLM-based language detection (closed-world against doc directory names)
    llm = None
    print("Using LLM for language detection...")
    llm_cfg = config.get("llm_retrieval", {})
    lang_detect_retries = _coerce_numeric(llm_cfg.get("lang_detect_retries", 2), default=2, minimum=0)
    lang_detect_backoff_sec = _coerce_float(
        llm_cfg.get("lang_detect_backoff_sec", 0.5), default=0.5, minimum=0.0
    )
    lang_detect_candidate_k = _coerce_numeric(
        llm_cfg.get("lang_detect_candidate_k", 12), default=12, minimum=2
    )
    lang_detect_card_chars = _coerce_numeric(
        llm_cfg.get("lang_detect_card_chars", 220), default=220, minimum=80
    )
    lang_detect_global_card_chars = _coerce_numeric(
        llm_cfg.get("lang_detect_global_card_chars", 90), default=90, minimum=60
    )

    try:
        if not known_languages:
            print("No language directories found; skipping language detection.")
            raise ValueError("known_languages_empty")

        llm = get_llm("retrieval")
        allowed_languages_json = json.dumps(sorted(known_languages), ensure_ascii=False)
        all_lang_ids = sorted(known_languages)
        global_cards = build_language_cards(
            doc_dir,
            all_lang_ids,
            max_chars=lang_detect_global_card_chars,
        )
        all_language_cards = [
            {"id": lang_id, "card": global_cards.get(lang_id, "")}
            for lang_id in all_lang_ids
        ]
        all_language_cards_json = json.dumps(all_language_cards, ensure_ascii=False)

        stage_a_prompt = ChatPromptTemplate.from_template(
            """Stage A (high-recall): identify candidate language IDs explicitly requested in the query.
You must select from this allowed language ID set only:
{allowed_languages_json}

Allowed language cards (for grounding names/transliterations):
{all_language_cards_json}

Return JSON only, in this schema:
{{"languages": ["lang_id_1", "lang_id_2", "..."]}}

Rules:
1) Include every explicitly mentioned language.
2) Never output values outside the allowed set.
3) If uncertain between near alternatives, include both candidates.
4) Return at most {candidate_k} IDs.
5) If no language is explicit, return {{"languages": []}}.
6) No prose, no markdown, JSON only.

Query: {query}
JSON:"""
        )

        stage_a_chain = stage_a_prompt | llm | StrOutputParser()
        stage_a_result = _invoke_with_retry(
            stage_a_chain,
            {
                "query": query,
                "allowed_languages_json": allowed_languages_json,
                "all_language_cards_json": all_language_cards_json,
                "candidate_k": lang_detect_candidate_k,
            },
            retries=lang_detect_retries,
            backoff_sec=lang_detect_backoff_sec,
            diagnostics=diagnostics,
            timeout_location="metadata_detection_stage_a",
        )
        diagnostics["lang_detect_method"] = "llm"
        diagnostics["lang_detect_stage"] = "stage_a"
        print(f"LLM Detected Language (stage_a): {stage_a_result}")

        stage_a_parsed = parse_detected_languages(
            stage_a_result,
            known_languages,
            max_languages=lang_detect_candidate_k,
        )
        candidate_langs = _as_lang_list(stage_a_parsed)
        diagnostics["detected_languages_candidates"] = candidate_langs

        final_parsed = None
        if candidate_langs:
            language_cards = build_language_cards(
                doc_dir,
                candidate_langs,
                max_chars=lang_detect_card_chars,
            )
            candidate_cards = [
                {"id": lang_id, "card": language_cards.get(lang_id, "")}
                for lang_id in candidate_langs
            ]
            candidate_cards_json = json.dumps(candidate_cards, ensure_ascii=False)

            stage_b_prompt = ChatPromptTemplate.from_template(
                """Stage B (precision): choose final language IDs explicitly requested in the query.
Select only from these candidate IDs:
{candidate_langs_json}

Candidate language cards:
{candidate_cards_json}

Return JSON only:
{{"languages": ["lang_id_1", "lang_id_2"]}}

Rules:
1) Use only candidate IDs.
2) Include only explicitly requested languages.
3) Keep query mention order.
4) If no explicit language, return {{"languages": []}}.
5) No prose, no markdown, JSON only.

Query: {query}
JSON:"""
            )

            stage_b_chain = stage_b_prompt | llm | StrOutputParser()
            stage_b_result = _invoke_with_retry(
                stage_b_chain,
                {
                    "query": query,
                    "candidate_langs_json": json.dumps(candidate_langs, ensure_ascii=False),
                    "candidate_cards_json": candidate_cards_json,
                },
                retries=lang_detect_retries,
                backoff_sec=lang_detect_backoff_sec,
                diagnostics=diagnostics,
                timeout_location="metadata_detection_stage_b",
            )
            diagnostics["lang_detect_stage"] = "stage_b"
            print(f"LLM Detected Language (stage_b): {stage_b_result}")
            final_parsed = parse_detected_languages(stage_b_result, known_languages, max_languages=3)

        if final_parsed is None and 1 <= len(candidate_langs) <= 3:
            final_parsed = candidate_langs if len(candidate_langs) > 1 else candidate_langs[0]

        if isinstance(final_parsed, str):
            diagnostics["detected_languages"] = [final_parsed]
        elif isinstance(final_parsed, list):
            diagnostics["detected_languages"] = final_parsed
            print(f"Multi-language query detected: {final_parsed}")
        metadata["lang"] = final_parsed
    except ValueError as e:
        if str(e) != "known_languages_empty":
            print(f"Metadata extraction failed: {e}")
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        if _is_timeout_exception(e) and diagnostics["timeout_location"] == "none":
            diagnostics["timeout_location"] = "metadata_detection"

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

    print(
        "LANG_DETECT_DIAGNOSTICS "
        f"lang_detect_method={diagnostics['lang_detect_method']} "
        f"lang_detect_stage={diagnostics.get('lang_detect_stage', 'none')} "
        f"detected_languages_candidates={diagnostics.get('detected_languages_candidates', [])} "
        f"detected_languages={diagnostics['detected_languages']} "
        f"enable_language_filter={diagnostics.get('enable_language_filter', True)} "
        f"viking_use_detected_language={diagnostics.get('viking_use_detected_language', True)} "
        f"viking_language_seeded={diagnostics.get('viking_language_seeded', False)} "
        f"l0_only={diagnostics.get('l0_only', False)} "
        f"filter_applied={diagnostics['filter_applied']} "
        f"timeout_location={diagnostics['timeout_location']}"
    )

    return metadata


class RAGRetriever:
    def __init__(self):
        self.config = src.utils.load_config()  # Use module-level for patching support
        self.db_cfg = self.config.get("database", {})
        self.backend_mode = str(self.db_cfg.get("type", "postgres")).lower()
        if self.backend_mode != "postgres":
            raise ValueError(
                f"Unsupported database.type='{self.backend_mode}'. "
                "Current retriever backend supports postgres only."
            )
        
        self.embedding_model_name = self.config["embedding"]["model_name"]
        configured_device = self.config.get("embedding", {}).get("device", "auto")
        self.device = src.utils.resolve_torch_device(configured_device)
        if (configured_device or "auto").lower() != self.device:
            print(f"Device fallback: embedding.device='{configured_device}' -> '{self.device}'")
        
        self.reranker_model_name = self.config["reranker"]["model_name"]
        self.use_reranker = self.config["reranker"]["enabled"]
        self.top_n = self.config["reranker"]["top_n"]
        reranker_cfg = self.config.get("reranker", {})
        self.pre_rerank_cap = max(0, int(reranker_cfg.get("pre_rerank_cap", 300)))
        self.rerank_max_input_chars = max(256, int(reranker_cfg.get("max_input_chars", 4000)))
        
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
        self.enable_language_filter = bool(retrieval_cfg.get("enable_language_filter", True))
        self.viking_use_detected_language = bool(retrieval_cfg.get("viking_use_detected_language", True))
        self.l0_only = bool(retrieval_cfg.get("l0_only", False))
        self.vision_enabled = retrieval_cfg.get("vision_search", False)
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

    def _bm25_lang_filter(self, bm25_results, lang_filter_value):
        """Apply metadata->>'lang' filter to BM25 results before RRF merge."""
        if not bm25_results or not lang_filter_value:
            return bm25_results

        requested_langs = (
            [lang_filter_value] if isinstance(lang_filter_value, str)
            else [str(v).strip() for v in lang_filter_value if str(v).strip()]
        )
        if not requested_langs:
            return bm25_results

        doc_ids = [doc_id for doc_id, _ in bm25_results]
        id_placeholders = ",".join(["%s"] * len(doc_ids))
        lang_placeholders = ",".join(["%s"] * len(requested_langs))

        query_sql = sql.SQL(
            """
            SELECT id FROM {}
            WHERE id IN ({}) AND metadata->>'lang' IN ({})
            """
        ).format(
            sql.Identifier(self.table_name),
            sql.SQL(id_placeholders),
            sql.SQL(lang_placeholders),
        )
        params = tuple(doc_ids + requested_langs)

        valid_ids = set()
        with self.conn.cursor() as cur:
            cur.execute(query_sql, params)
            for row in cur.fetchall():
                valid_ids.add(row[0])

        filtered = [(doc_id, score) for doc_id, score in bm25_results if doc_id in valid_ids]
        if len(filtered) < len(bm25_results):
            print(f"BM25 Lang Filter: {len(bm25_results)} → {len(filtered)} (langs={requested_langs})")
        return filtered

    def _bm25_l0_filter(self, bm25_results):
        """Apply metadata->>'level'='0' filter to BM25 results before RRF merge."""
        if not bm25_results:
            return bm25_results

        doc_ids = [doc_id for doc_id, _ in bm25_results]
        id_placeholders = ",".join(["%s"] * len(doc_ids))
        query_sql = sql.SQL(
            """
            SELECT id FROM {}
            WHERE id IN ({}) AND metadata->>'level' = '0'
            """
        ).format(
            sql.Identifier(self.table_name),
            sql.SQL(id_placeholders),
        )

        valid_ids = set()
        with self.conn.cursor() as cur:
            cur.execute(query_sql, tuple(doc_ids))
            for row in cur.fetchall():
                valid_ids.add(row[0])

        filtered = [(doc_id, score) for doc_id, score in bm25_results if doc_id in valid_ids]
        if len(filtered) < len(bm25_results):
            print(f"BM25 L0 Filter: {len(bm25_results)} → {len(filtered)}")
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
        
        if metadata is None:
            metadata = {}
        diagnostics = metadata.get("_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {
                "lang_detect_method": "none",
                "detected_languages": [],
                "detected_languages_candidates": [],
                "lang_detect_stage": "none",
                "enable_language_filter": self.enable_language_filter,
                "viking_use_detected_language": self.viking_use_detected_language,
                "viking_language_seeded": False,
                "l0_only": self.l0_only,
                "filter_applied": False,
                "timeout_location": "none",
            }
            metadata["_diagnostics"] = diagnostics
        diagnostics["enable_language_filter"] = self.enable_language_filter
        diagnostics["viking_use_detected_language"] = self.viking_use_detected_language
        diagnostics["l0_only"] = self.l0_only
        diagnostics["filter_applied"] = False

        # Build language filter clause
        lang_filter = ""
        lang_param = []
        if self.enable_language_filter and metadata and metadata.get('lang'):
            lang = metadata['lang']
            if isinstance(lang, list):
                # Multi-language query - use IN clause
                placeholders = ','.join(['%s'] * len(lang))
                lang_filter = f" AND metadata->>'lang' IN ({placeholders})"
                lang_param = lang
                diagnostics["filter_applied"] = True
                print(f"Applying Multi-Language Filter: {lang}")
            else:
                # Single language
                print(f"Applying Language Filter: {lang}")
                lang_filter = " AND metadata->>'lang' = %s"
                lang_param = [lang]
                diagnostics["filter_applied"] = True
        elif metadata and metadata.get("lang") and not self.enable_language_filter:
            print("Language Filter: disabled by retrieval.enable_language_filter=false")

        level_filter = ""
        if self.l0_only:
            level_filter = " AND metadata->>'level' = '0'"
            print("Applying L0-only Filter: metadata.level=0")

        print(
            "LANG_FILTER_DIAGNOSTICS "
            f"lang_detect_method={diagnostics.get('lang_detect_method', 'none')} "
            f"lang_detect_stage={diagnostics.get('lang_detect_stage', 'none')} "
            f"detected_languages_candidates={diagnostics.get('detected_languages_candidates', [])} "
            f"detected_languages={diagnostics.get('detected_languages', [])} "
            f"enable_language_filter={diagnostics.get('enable_language_filter', True)} "
            f"viking_use_detected_language={diagnostics.get('viking_use_detected_language', True)} "
            f"viking_language_seeded={diagnostics.get('viking_language_seeded', False)} "
            f"l0_only={diagnostics.get('l0_only', False)} "
            f"filter_applied={diagnostics.get('filter_applied', False)} "
            f"timeout_location={diagnostics.get('timeout_location', 'none')}"
        )
        
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
                WHERE (embedding <=> %s::vector) < %s {} {} {}
                ORDER BY distance ASC LIMIT %s
                """
            ).format(
                sql.Identifier(self.table_name),
                sql.SQL(lang_filter),
                sql.SQL(level_filter),
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
                if self.l0_only and bm25_results:
                    bm25_results = self._bm25_l0_filter(bm25_results)
                if self.enable_language_filter and metadata and metadata.get('lang'):
                    bm25_results = self._bm25_lang_filter(bm25_results, metadata.get('lang'))
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
            lang_filter = metadata.get('lang') if self.enable_language_filter else None
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

        original_count = len(docs)
        candidates = docs

        if self.pre_rerank_cap > 0 and len(candidates) > self.pre_rerank_cap:
            candidates = sorted(
                candidates,
                key=lambda d: d.metadata.get("_vector_rank", 10**9),
            )[:self.pre_rerank_cap]
            print(
                f"Pre-rerank cap applied: {original_count} -> {len(candidates)} "
                f"(cap={self.pre_rerank_cap})"
            )

        # Preserve vector top-5 (using _vector_rank metadata)
        vector_top = [d for d in candidates if d.metadata.get('_vector_rank', 999) < 5]

        print(f"Loading Reranker ({self.reranker_model_name})...")
        reranker = CrossEncoder(
            self.reranker_model_name, 
            device=self.device,
            trust_remote_code=True
        )

        pairs = []
        truncated_count = 0
        for doc in candidates:
            content = doc.page_content or ""
            if len(content) > self.rerank_max_input_chars:
                content = content[:self.rerank_max_input_chars]
                truncated_count += 1
            pairs.append([query, content])

        if truncated_count > 0:
            print(
                f"Rerank input truncation: {truncated_count}/{len(candidates)} docs "
                f"truncated to {self.rerank_max_input_chars} chars"
            )

        scores = reranker.predict(pairs, batch_size=4)
        
        del reranker
        self._clean_memory()
        print("Reranker unloaded.")
        
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
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
        
        print(
            f"Reranked: Reduced from {original_count} to {len(top_n_docs)} docs "
            f"(candidate={len(candidates)}, Top-N={top_n})"
        )
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

        diagnostics = metadata.get("_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {
                "lang_detect_method": "none",
                "detected_languages": [],
                "detected_languages_candidates": [],
                "lang_detect_stage": "none",
                "enable_language_filter": self.enable_language_filter,
                "viking_use_detected_language": self.viking_use_detected_language,
                "viking_language_seeded": False,
                "l0_only": self.l0_only,
                "filter_applied": False,
                "timeout_location": "none",
            }
            metadata["_diagnostics"] = diagnostics
        diagnostics["enable_language_filter"] = self.enable_language_filter
        diagnostics["viking_use_detected_language"] = self.viking_use_detected_language
        diagnostics["l0_only"] = self.l0_only

        # Viking routing (search-space restriction)
        viking_scope = None
        if self.viking_enabled:
            from src.retrieve.viking_router import compute_scope
            from src.retrieve.viking_index import get_taxonomy_index
            data_path = self.config["project"]["data_path"]
            tax_index = get_taxonomy_index(data_path)
            viking_metadata = dict(metadata)
            if not self.viking_use_detected_language:
                viking_metadata.pop("lang", None)
                viking_metadata["_viking_use_detected_language"] = False
                diagnostics["viking_language_seeded"] = False
            else:
                viking_metadata["_viking_use_detected_language"] = True
                diagnostics["viking_language_seeded"] = bool(metadata.get("lang"))
            viking_scope = compute_scope(query, viking_metadata, tax_index)
            print(f"Viking Scope: conf={viking_scope.confidence}, "
                  f"patterns={len(viking_scope.path_patterns)}, "
                  f"trace={viking_scope.trace}")
        else:
            diagnostics["viking_language_seeded"] = False

        # 1. Text Retrieval (Sequential)
        text_docs = self.retrieve_text(query, k, metadata=metadata,
                                       viking_scope=viking_scope)
        print(f"Text Matches: {len(text_docs)}")
        print(
            "RETRIEVAL_DIAGNOSTICS "
            f"lang_detect_method={diagnostics.get('lang_detect_method', 'none')} "
            f"lang_detect_stage={diagnostics.get('lang_detect_stage', 'none')} "
            f"detected_languages_candidates={diagnostics.get('detected_languages_candidates', [])} "
            f"detected_languages={diagnostics.get('detected_languages', [])} "
            f"enable_language_filter={diagnostics.get('enable_language_filter', True)} "
            f"viking_use_detected_language={diagnostics.get('viking_use_detected_language', True)} "
            f"viking_language_seeded={diagnostics.get('viking_language_seeded', False)} "
            f"l0_only={diagnostics.get('l0_only', False)} "
            f"filter_applied={diagnostics.get('filter_applied', False)} "
            f"timeout_location={diagnostics.get('timeout_location', 'none')}"
        )

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
