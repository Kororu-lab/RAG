#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent.graph import app as rag_app
from src.eval.metrics import aggregate_macro_micro, build_gold_sets, compute_query_metrics
from src.eval.profiles import (
    RERANK_K30_PROFILES,
    RERANK_MIN_TOP_N_FOR_K30,
    load_profile_file,
    resolve_profile_config,
    select_profiles,
)
from src.llm.factory import get_llm
from src.llm.json_retry import parse_json_strict, repair_json_text
from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.retrieve.split_budget import compute_split_budgets
from src.utils import load_config, use_config_override


K_VALUES = (5, 10, 20, 30)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_timeout_exception(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "timeout" in msg
        or "read timed out" in msg
        or "timed out" in msg
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_qaset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query/QA file not found: {path}")

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    rows.append(
                        {
                            "query_id": f"Q{line_no:04d}",
                            "query_type": "unknown",
                            "query": obj,
                            "expected_languages": [],
                            "expected_topics": [],
                            "gold_docs": [],
                            "gold_chunks": [],
                        }
                    )
                    continue

                query = obj.get("query") or obj.get("question") or obj.get("text")
                if not query:
                    continue
                rows.append(
                    {
                        "query_id": str(obj.get("query_id") or obj.get("id") or f"Q{line_no:04d}"),
                        "query_type": str(obj.get("query_type") or "unknown"),
                        "query": str(query),
                        "expected_languages": obj.get("expected_languages") or [],
                        "expected_topics": obj.get("expected_topics") or [],
                        "gold_docs": obj.get("gold_docs") or [],
                        "gold_chunks": obj.get("gold_chunks") or [],
                    }
                )
        return rows

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("JSON input must be a list of QA objects or {'items': [...]}.")

    for i, obj in enumerate(items, start=1):
        if not isinstance(obj, dict):
            continue
        query = obj.get("query") or obj.get("question") or obj.get("text")
        if not query:
            continue
        rows.append(
            {
                "query_id": str(obj.get("query_id") or obj.get("id") or f"Q{i:04d}"),
                "query_type": str(obj.get("query_type") or "unknown"),
                "query": str(query),
                "expected_languages": obj.get("expected_languages") or [],
                "expected_topics": obj.get("expected_topics") or [],
                "gold_docs": obj.get("gold_docs") or [],
                "gold_chunks": obj.get("gold_chunks") or [],
            }
        )

    return rows


def _doc_key(meta: Dict[str, Any], rank: int) -> str:
    source_file = meta.get("source_file")
    chunk_id = meta.get("chunk_id")
    if source_file is not None and chunk_id is not None:
        return f"{source_file}:{chunk_id}"
    if source_file is not None:
        return f"{source_file}:?"
    ref_id = meta.get("ref_id")
    if ref_id:
        return str(ref_id)
    db_id = meta.get("_db_id")
    if db_id is not None:
        return f"db:{db_id}"
    return f"rank:{rank}"


def _doc_identity(doc: Document) -> Tuple[str, str]:
    source_file = str(doc.metadata.get("source_file", "")).strip()
    chunk_id = doc.metadata.get("chunk_id")
    if source_file:
        return source_file, str(chunk_id) if chunk_id is not None else ""
    ref_id = str(doc.metadata.get("ref_id", "")).strip()
    if ref_id:
        return "ref", ref_id
    db_id = doc.metadata.get("_db_id")
    if db_id is not None:
        return "db", str(db_id)
    return "obj", str(id(doc))


def _doc_score(doc: Document) -> float:
    try:
        score = doc.metadata.get("score")
        if score is None:
            return float("-inf")
        return float(score)
    except Exception:
        return float("-inf")


def _merge_split_docs(
    *,
    branch_docs: Dict[str, List[Document]],
    language_order: List[str],
    merge_cap: int,
) -> List[Document]:
    prepared: Dict[str, List[Document]] = {}
    for lang in language_order:
        docs = list(branch_docs.get(lang, []))
        docs.sort(key=_doc_score, reverse=True)
        prepared[lang] = docs

    merged: List[Document] = []
    seen = set()
    cursor = {lang: 0 for lang in language_order}
    cap = max(0, int(merge_cap))
    if cap <= 0:
        return []

    while len(merged) < cap:
        progressed = False
        for lang in language_order:
            docs = prepared.get(lang, [])
            idx = cursor[lang]
            while idx < len(docs):
                candidate = docs[idx]
                idx += 1
                key = _doc_identity(candidate)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
                progressed = True
                break
            cursor[lang] = idx
            if len(merged) >= cap:
                break
        if not progressed:
            break

    return merged


def _generate_split_queries(
    *,
    question: str,
    languages: List[str],
    retries: int,
    backoff_sec: float,
    json_retries: int,
    json_backoff_sec: float,
) -> tuple[Dict[str, str], int]:
    llm = get_llm("retrieval")
    prompt = ChatPromptTemplate.from_template(
        """Rewrite the user query into one focused mini-query per target language ID.
Return JSON only in this exact schema:
{{"queries": {{"lang_id_1": "query text", "lang_id_2": "query text"}}}}

Rules:
1) Keep the original user intent/topic.
2) Each mini-query must focus on only the specified language ID.
3) Use only these target language IDs: {languages_json}
4) Every target language ID must appear exactly once in "queries".
5) No prose, no markdown, JSON only.

User query: {question}
JSON:"""
    )
    chain = prompt | llm | StrOutputParser()

    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    json_retry_count = 0
    for attempt in range(1, attempts + 1):
        try:
            raw = (chain.invoke(
                {
                    "question": question,
                    "languages_json": json.dumps(languages, ensure_ascii=False),
                }
            ) or "").strip()
            try:
                parsed = parse_json_strict(raw)
            except Exception:
                repaired_raw, retries_used, repaired_ok = repair_json_text(
                    llm=llm,
                    raw_text=raw,
                    schema_hint='{"queries": {"lang_id_1": "query text", "lang_id_2": "query text"}}',
                    retries=max(0, int(json_retries)),
                    backoff_sec=max(0.0, float(json_backoff_sec)),
                    timeout_checker=_is_timeout_exception,
                    on_timeout=None,
                    timeout_location="split_query_json_repair",
                )
                json_retry_count += retries_used
                if not repaired_ok:
                    raise ValueError("split_query_json_invalid")
                parsed = parse_json_strict(repaired_raw)
            queries = parsed.get("queries", {}) if isinstance(parsed, dict) else {}
            if not isinstance(queries, dict):
                raise ValueError("split_query_invalid_schema")

            out: Dict[str, str] = {}
            for lang in languages:
                mini_query = queries.get(lang)
                if not isinstance(mini_query, str) or not mini_query.strip():
                    raise ValueError(f"split_query_missing_lang:{lang}")
                out[lang] = mini_query.strip()
            return out, json_retry_count
        except Exception as e:
            last_error = e
            if attempt >= attempts:
                break
            wait_sec = max(0.0, backoff_sec) * attempt
            if wait_sec > 0:
                sleep(wait_sec)

    raise RuntimeError(f"split_query_generation_failed: {last_error}")


def _documents_to_ranked_rows(docs: Sequence[Document]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, doc in enumerate(docs, start=1):
        meta = deepcopy(doc.metadata or {})
        rows.append(
            {
                "rank": rank,
                "doc_key": _doc_key(meta, rank),
                "db_id": meta.get("_db_id"),
                "source_file": meta.get("source_file"),
                "chunk_id": meta.get("chunk_id"),
                "level": meta.get("level"),
                "lang": meta.get("lang"),
                "ref_id": meta.get("ref_id"),
                "score": meta.get("score"),
                "vector_rank": meta.get("_vector_rank"),
            }
        )
    return rows


def _extract_detected_languages(metadata: Dict[str, Any]) -> List[str]:
    diagnostics = metadata.get("_diagnostics", {}) if isinstance(metadata, dict) else {}
    if isinstance(metadata.get("lang"), list):
        langs = [str(v).strip() for v in metadata.get("lang", []) if str(v).strip()]
        if langs:
            return langs
    return [str(v).strip() for v in diagnostics.get("detected_languages", []) if str(v).strip()]


def _retrieve_with_optional_split(
    *,
    retriever: RAGRetriever,
    query_text: str,
    profile_cfg: Dict[str, Any],
    fixed_k: int,
    fixed_top_n: int,
) -> Tuple[Dict[str, Any], List[Document], List[str], float, float]:
    t_meta = perf_counter()
    metadata = extract_query_metadata(query_text, profile_cfg)
    metadata_elapsed_sec = perf_counter() - t_meta
    diagnostics = metadata.setdefault("_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        metadata["_diagnostics"] = diagnostics

    diagnostics.setdefault("lang_detect_method", "none")
    diagnostics.setdefault("detected_languages", [])
    diagnostics.setdefault("filter_applied", False)
    diagnostics.setdefault("split_applied", False)
    diagnostics.setdefault("split_branches", [])
    diagnostics.setdefault("timeout_location", "none")
    diagnostics.setdefault("lang_json_retry_count", 0)
    diagnostics.setdefault("lang_json_valid_stage_a", False)
    diagnostics.setdefault("lang_json_valid_stage_b", False)
    diagnostics.setdefault("lang_json_fail_open", False)
    diagnostics.setdefault("split_json_retry_count", 0)
    diagnostics.setdefault("split_json_fail_open", False)

    warnings: List[str] = []

    retrieval_cfg = profile_cfg.get("retrieval", {})
    split_cfg = retrieval_cfg.get("query_split", {}) or {}
    split_enabled = bool(split_cfg.get("enabled", False))
    branch_mode = str(split_cfg.get("branch_k_mode", "full_per_branch"))
    max_languages = max(2, _safe_int(split_cfg.get("max_languages", 3), 3))

    detected_languages = _extract_detected_languages(metadata)

    if split_enabled and branch_mode in {"balanced", "full_per_branch"} and len(detected_languages) >= 2:
        languages = detected_languages[:max_languages]
        llm_cfg = profile_cfg.get("llm_retrieval", {})
        retries = max(0, _safe_int(llm_cfg.get("lang_detect_retries", 2), 2))
        backoff_sec = max(0.0, _safe_float(llm_cfg.get("lang_detect_backoff_sec", 0.5), 0.5))
        json_retries = max(0, _safe_int(llm_cfg.get("lang_json_retries", 1), 1))
        json_backoff_sec = max(0.0, _safe_float(llm_cfg.get("lang_json_backoff_sec", 0.25), 0.25))

        try:
            t_retrieve = perf_counter()
            split_queries, split_json_retry_count = _generate_split_queries(
                question=query_text,
                languages=languages,
                retries=retries,
                backoff_sec=backoff_sec,
                json_retries=json_retries,
                json_backoff_sec=json_backoff_sec,
            )
            budgets = compute_split_budgets(
                total_k=fixed_k,
                total_top_n=fixed_top_n,
                branch_count=len(languages),
                mode=branch_mode,
            )

            branch_docs: Dict[str, List[Document]] = {}
            for idx, lang in enumerate(languages):
                branch_k = budgets.k_allocs[idx]
                if branch_k <= 0:
                    continue
                branch_top_n = max(1, budgets.top_n_allocs[idx])
                branch_metadata = deepcopy(metadata)
                branch_metadata["lang"] = lang
                branch_diag = branch_metadata.setdefault("_diagnostics", {})
                branch_diag["split_applied"] = True
                branch_diag["split_branches"] = languages

                docs = retriever.retrieve_documents(
                    split_queries[lang],
                    metadata=branch_metadata,
                    k=branch_k,
                    top_n=branch_top_n,
                )
                branch_docs[lang] = docs

            merged_docs = _merge_split_docs(
                branch_docs=branch_docs,
                language_order=languages,
                merge_cap=budgets.merge_cap,
            )
            diagnostics["split_applied"] = True
            diagnostics["split_branches"] = languages
            diagnostics["detected_languages"] = languages
            diagnostics["filter_applied"] = True
            diagnostics["split_json_retry_count"] = int(split_json_retry_count)
            diagnostics["split_json_fail_open"] = False
            retrieval_elapsed_sec = perf_counter() - t_retrieve
            return metadata, merged_docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec
        except Exception as e:
            diagnostics["split_applied"] = False
            diagnostics["split_branches"] = []
            diagnostics["split_json_fail_open"] = True
            warnings.append(
                "Split-query generation/retrieval failed; fallback to one-shot multi-language retrieval."
            )
            if "timeout" in str(e).lower() and diagnostics.get("timeout_location", "none") == "none":
                diagnostics["timeout_location"] = "split_query_generation"

    t_retrieve = perf_counter()
    docs = retriever.retrieve_documents(
        query_text,
        metadata=metadata,
        k=fixed_k,
        top_n=fixed_top_n,
    )
    retrieval_elapsed_sec = perf_counter() - t_retrieve
    return metadata, docs, warnings, metadata_elapsed_sec, retrieval_elapsed_sec


def _build_query_row_base(
    *,
    run_id: str,
    profile: str,
    query_item: Dict[str, Any],
    backend_mode_effective: str,
    backend_mode_config: str,
    backend_mode_required: str,
    fixed_k: int,
    fixed_top_n: int,
    retrieval_timeout_sec_effective: int,
    metadata_elapsed_sec: float,
    retrieval_elapsed_sec: float,
    total_elapsed_sec: float,
    diagnostics: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "profile": profile,
        "query_id": query_item["query_id"],
        "query_type": query_item.get("query_type", "unknown"),
        "query_text": query_item["query"],
        "timestamp": _utc_now(),
        "backend_mode": backend_mode_effective,
        "backend_mode_effective": backend_mode_effective,
        "backend_mode_config": backend_mode_config,
        "backend_mode_required": backend_mode_required,
        "k": fixed_k,
        "top_n": fixed_top_n,
        "retrieval_timeout_sec_effective": retrieval_timeout_sec_effective,
        "metadata_elapsed_sec": round(metadata_elapsed_sec, 6),
        "retrieval_elapsed_sec": round(retrieval_elapsed_sec, 6),
        "total_elapsed_sec": round(total_elapsed_sec, 6),
        "lang_detect_method": diagnostics.get("lang_detect_method", "none"),
        "detected_languages": diagnostics.get("detected_languages", []),
        "lang_json_retry_count": int(diagnostics.get("lang_json_retry_count", 0)),
        "lang_json_fail_open": bool(diagnostics.get("lang_json_fail_open", False)),
        "lang_json_valid_stage_a": bool(diagnostics.get("lang_json_valid_stage_a", False)),
        "lang_json_valid_stage_b": bool(diagnostics.get("lang_json_valid_stage_b", False)),
        "filter_applied": bool(diagnostics.get("filter_applied", False)),
        "split_applied": bool(diagnostics.get("split_applied", False)),
        "split_branches": diagnostics.get("split_branches", []),
        "split_json_retry_count": int(diagnostics.get("split_json_retry_count", 0)),
        "split_json_fail_open": bool(diagnostics.get("split_json_fail_open", False)),
        "timeout_location": diagnostics.get("timeout_location", "none"),
        "warnings": warnings,
    }


def _run_retrieval_profile(
    *,
    run_id: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    profile_dir: Path,
) -> Dict[str, Any]:
    retrieval_cfg = profile_cfg.get("retrieval", {})
    reranker_cfg = profile_cfg.get("reranker", {})
    ablation_cfg = profile_cfg.get("ablation", {})

    fixed_k = _safe_int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 30)), 30)
    fixed_top_n = _safe_int(retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 30)), 30)
    llm_retrieval_cfg = profile_cfg.get("llm_retrieval", {}) or {}
    retrieval_timeout_sec_effective = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
    if retrieval_timeout_sec_effective <= 0:
        retrieval_timeout_sec_effective = 600

    if profile_name in RERANK_K30_PROFILES and bool(reranker_cfg.get("enabled", False)) and fixed_top_n < RERANK_MIN_TOP_N_FOR_K30:
        raise RuntimeError(
            f"{profile_name} invalid: reranker enabled but fixed_top_n={fixed_top_n} < {RERANK_MIN_TOP_N_FOR_K30}."
        )

    required_backend_mode = str(ablation_cfg.get("required_backend_mode", "any")).lower()
    backend_mode_config = str((profile_cfg.get("database", {}) or {}).get("type", "postgres")).lower()

    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    with use_config_override(profile_cfg):
        with RAGRetriever() as retriever:
            backend_mode_effective = str(getattr(retriever, "backend_mode", "unknown")).lower()
            if required_backend_mode != "any" and backend_mode_effective != required_backend_mode:
                raise RuntimeError(
                    f"Backend mode mismatch: required={required_backend_mode}, actual={backend_mode_effective}"
                )

            for query_item in queries:
                query_text = query_item["query"]
                t_start = perf_counter()
                metadata, docs, split_warnings, metadata_elapsed_sec, retrieval_elapsed_sec = _retrieve_with_optional_split(
                    retriever=retriever,
                    query_text=query_text,
                    profile_cfg=profile_cfg,
                    fixed_k=fixed_k,
                    fixed_top_n=fixed_top_n,
                )
                total_elapsed_sec = perf_counter() - t_start

                diagnostics = metadata.get("_diagnostics", {}) if isinstance(metadata, dict) else {}
                ranked_rows = _documents_to_ranked_rows(docs)
                gold_docs, gold_chunks = build_gold_sets(query_item)
                metric_values = compute_query_metrics(
                    ranked_docs=ranked_rows,
                    gold_docs=gold_docs,
                    gold_chunks=gold_chunks,
                    ks=K_VALUES,
                )

                warnings = list(split_warnings)
                if diagnostics.get("timeout_location", "none") != "none":
                    warnings.append(
                        f"Language/split timeout at {diagnostics.get('timeout_location')}; fail-open retrieval continued."
                    )

                row_base = _build_query_row_base(
                    run_id=run_id,
                    profile=profile_name,
                    query_item=query_item,
                    backend_mode_effective=backend_mode_effective,
                    backend_mode_config=backend_mode_config,
                    backend_mode_required=required_backend_mode,
                    fixed_k=fixed_k,
                    fixed_top_n=fixed_top_n,
                    retrieval_timeout_sec_effective=retrieval_timeout_sec_effective,
                    metadata_elapsed_sec=metadata_elapsed_sec,
                    retrieval_elapsed_sec=retrieval_elapsed_sec,
                    total_elapsed_sec=total_elapsed_sec,
                    diagnostics=diagnostics,
                    warnings=warnings,
                )

                trace_row = dict(row_base)
                trace_row["documents"] = ranked_rows
                trace_row["doc_count"] = len(ranked_rows)
                trace_row["retrieval_diagnostics"] = diagnostics
                trace_row.update(metric_values)
                trace_rows.append(trace_row)

                summary_row = dict(row_base)
                summary_row["doc_count"] = len(ranked_rows)
                summary_row.update(metric_values)
                summary_rows.append(summary_row)

    _write_jsonl(profile_dir / "retrieval_traces.jsonl", trace_rows)
    _write_csv(profile_dir / "summary.csv", summary_rows)
    _write_json(profile_dir / "resolved_config.json", profile_cfg)

    macro_micro_rows = aggregate_macro_micro(
        rows=summary_rows,
        ks=K_VALUES,
        profile=profile_name,
        track="retrieval",
    )
    _write_csv(profile_dir / "macro_micro.csv", macro_micro_rows)

    return {
        "profile": profile_name,
        "track": "retrieval",
        "records": len(summary_rows),
        "backend_mode_effective": backend_mode_effective,
        "backend_mode_config": backend_mode_config,
        "backend_mode_required": required_backend_mode,
        "config_hash": _config_hash(profile_cfg),
        "macro_micro_rows": macro_micro_rows,
    }


def _run_e2e_profile(
    *,
    run_id: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    profile_dir: Path,
) -> Dict[str, Any]:
    retrieval_cfg = profile_cfg.get("retrieval", {})
    reranker_cfg = profile_cfg.get("reranker", {})
    ablation_cfg = profile_cfg.get("ablation", {})

    fixed_k = _safe_int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 30)), 30)
    fixed_top_n = _safe_int(retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 30)), 30)
    llm_retrieval_cfg = profile_cfg.get("llm_retrieval", {}) or {}
    retrieval_timeout_sec_effective = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
    if retrieval_timeout_sec_effective <= 0:
        retrieval_timeout_sec_effective = 600

    if bool(reranker_cfg.get("enabled", False)) and fixed_top_n < RERANK_MIN_TOP_N_FOR_K30:
        raise RuntimeError(
            f"{profile_name} invalid: reranker enabled but fixed_top_n={fixed_top_n} < {RERANK_MIN_TOP_N_FOR_K30}."
        )

    required_backend_mode = str(ablation_cfg.get("required_backend_mode", "any")).lower()
    backend_mode_config = str((profile_cfg.get("database", {}) or {}).get("type", "postgres")).lower()

    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    first_metric_rows: List[Dict[str, Any]] = []
    final_metric_rows: List[Dict[str, Any]] = []

    with use_config_override(profile_cfg):
        for query_item in queries:
            query_text = query_item["query"]
            t_start = perf_counter()

            first_retrieve_docs: List[Document] = []
            first_diagnostics: Dict[str, Any] = {}
            final_docs: List[Document] = []
            warnings: List[str] = []
            timeout_locations: List[str] = []
            generation = ""
            generation_diagnostics: Dict[str, Any] = {}

            for event in rag_app.stream({"question": query_text, "search_count": 0}):
                for node_name, node_value in event.items():
                    if not isinstance(node_value, dict):
                        continue
                    if node_name == "retrieve" and not first_retrieve_docs:
                        first_retrieve_docs = list(node_value.get("documents") or [])
                        first_diagnostics = node_value.get("retrieval_diagnostics", {}) or {}
                    docs = node_value.get("documents")
                    if isinstance(docs, list):
                        final_docs = docs
                    if isinstance(node_value.get("warnings"), list):
                        warnings = list(node_value.get("warnings") or [])
                    if isinstance(node_value.get("timeout_locations"), list):
                        timeout_locations = list(node_value.get("timeout_locations") or [])
                    if isinstance(node_value.get("generation"), str):
                        generation = node_value.get("generation")
                    if isinstance(node_value.get("generation_diagnostics"), dict):
                        generation_diagnostics = dict(node_value.get("generation_diagnostics") or {})

            total_elapsed_sec = perf_counter() - t_start
            if required_backend_mode != "any" and backend_mode_config != required_backend_mode:
                raise RuntimeError(
                    f"Backend mode mismatch (E2E config): required={required_backend_mode}, config={backend_mode_config}"
                )

            first_ranked = _documents_to_ranked_rows(first_retrieve_docs)
            final_ranked = _documents_to_ranked_rows(final_docs)
            gold_docs, gold_chunks = build_gold_sets(query_item)
            first_metrics = compute_query_metrics(
                ranked_docs=first_ranked,
                gold_docs=gold_docs,
                gold_chunks=gold_chunks,
                ks=K_VALUES,
            )
            final_metrics = compute_query_metrics(
                ranked_docs=final_ranked,
                gold_docs=gold_docs,
                gold_chunks=gold_chunks,
                ks=K_VALUES,
            )

            diagnostics = first_diagnostics if isinstance(first_diagnostics, dict) else {}
            row_base = {
                "run_id": run_id,
                "profile": profile_name,
                "query_id": query_item["query_id"],
                "query_type": query_item.get("query_type", "unknown"),
                "query_text": query_text,
                "timestamp": _utc_now(),
                "backend_mode": backend_mode_config,
                "backend_mode_effective": backend_mode_config,
                "backend_mode_config": backend_mode_config,
                "backend_mode_required": required_backend_mode,
                "k": fixed_k,
                "top_n": fixed_top_n,
                "retrieval_timeout_sec_effective": retrieval_timeout_sec_effective,
                "total_elapsed_sec": round(total_elapsed_sec, 6),
                "lang_detect_method": diagnostics.get("lang_detect_method", "none"),
                "detected_languages": diagnostics.get("detected_languages", []),
                "lang_json_retry_count": int(diagnostics.get("lang_json_retry_count", 0)),
                "lang_json_fail_open": bool(diagnostics.get("lang_json_fail_open", False)),
                "lang_json_valid_stage_a": bool(diagnostics.get("lang_json_valid_stage_a", False)),
                "lang_json_valid_stage_b": bool(diagnostics.get("lang_json_valid_stage_b", False)),
                "filter_applied": bool(diagnostics.get("filter_applied", False)),
                "split_applied": bool(diagnostics.get("split_applied", False)),
                "split_branches": diagnostics.get("split_branches", []),
                "split_json_retry_count": int(diagnostics.get("split_json_retry_count", 0)),
                "split_json_fail_open": bool(diagnostics.get("split_json_fail_open", False)),
                "timeout_location": diagnostics.get("timeout_location", "none"),
                "warnings": warnings,
                "timeout_locations": timeout_locations,
                "first_doc_count": len(first_ranked),
                "final_doc_count": len(final_ranked),
                "generation": generation,
                "generation_diagnostics": generation_diagnostics,
                "generation_grounding_mode": generation_diagnostics.get("grounding_mode", "unknown"),
                "generation_inline_citation_coverage": float(generation_diagnostics.get("inline_citation_coverage", 0.0) or 0.0),
                "generation_guard_triggered": bool(generation_diagnostics.get("unsupported_claim_guard_triggered", False)),
                "generation_guard_reason": generation_diagnostics.get("fail_closed_reason", "none"),
            }

            trace_row = dict(row_base)
            trace_row["first_retrieve_documents"] = first_ranked
            trace_row["final_documents"] = final_ranked
            trace_row.update({f"first_{k}": v for k, v in first_metrics.items()})
            trace_row.update({f"final_{k}": v for k, v in final_metrics.items()})
            trace_rows.append(trace_row)

            summary_row = dict(row_base)
            summary_row.update({f"first_{k}": v for k, v in first_metrics.items()})
            summary_row.update({f"final_{k}": v for k, v in final_metrics.items()})
            summary_rows.append(summary_row)

            first_row = {
                "profile": profile_name,
                "query_id": query_item["query_id"],
                "query_type": query_item.get("query_type", "unknown"),
            }
            first_row.update(first_metrics)
            first_metric_rows.append(first_row)

            final_row = {
                "profile": profile_name,
                "query_id": query_item["query_id"],
                "query_type": query_item.get("query_type", "unknown"),
            }
            final_row.update(final_metrics)
            final_metric_rows.append(final_row)

    _write_jsonl(profile_dir / "e2e_traces.jsonl", trace_rows)
    _write_csv(profile_dir / "e2e_summary.csv", summary_rows)
    _write_json(profile_dir / "resolved_config.json", profile_cfg)

    first_macro = aggregate_macro_micro(
        rows=first_metric_rows,
        ks=K_VALUES,
        profile=profile_name,
        track="e2e_first_retrieve",
    )
    final_macro = aggregate_macro_micro(
        rows=final_metric_rows,
        ks=K_VALUES,
        profile=profile_name,
        track="e2e_final",
    )
    macro_rows = first_macro + final_macro
    _write_csv(profile_dir / "e2e_macro_micro.csv", macro_rows)

    return {
        "profile": profile_name,
        "track": "e2e",
        "records": len(summary_rows),
        "backend_mode_effective": backend_mode_config,
        "backend_mode_config": backend_mode_config,
        "backend_mode_required": required_backend_mode,
        "config_hash": _config_hash(profile_cfg),
        "macro_micro_rows": macro_rows,
    }


def run_profile(
    *,
    run_id: str,
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    profile_dir = out_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    run_mode = str((profile_cfg.get("ablation", {}) or {}).get("run_mode", "retrieval")).lower()
    if run_mode == "e2e":
        return _run_e2e_profile(
            run_id=run_id,
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            queries=queries,
            profile_dir=profile_dir,
        )

    return _run_retrieval_profile(
        run_id=run_id,
        profile_name=profile_name,
        profile_cfg=profile_cfg,
        queries=queries,
        profile_dir=profile_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0-B7 ablation harness.")
    parser.add_argument("--queries", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument(
        "--profile-config",
        default="config/ablation_profiles.yaml",
        help="Path to ablation profile config YAML.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names (default: all from config).",
    )
    parser.add_argument(
        "--outdir",
        default="eval/runs",
        help="Output base directory.",
    )
    parser.add_argument("--run-id", default="", help="Optional run id. Default UTC timestamp.")
    parser.add_argument("--fixed-k", type=int, default=None, help="Override fixed_k for all profiles.")
    parser.add_argument(
        "--fixed-top-n",
        type=int,
        default=None,
        help="Override fixed_top_n for all profiles.",
    )
    args = parser.parse_args()

    base_config = load_config()
    profile_path = Path(args.profile_config)
    defaults, profiles_map = load_profile_file(profile_path)
    selected_profiles = select_profiles(profiles_map, args.profiles)

    queries = _load_qaset(Path(args.queries))
    if not queries:
        raise ValueError("No queries loaded from input file.")

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.outdir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "query_file": str(Path(args.queries)),
        "query_count": len(queries),
        "profile_config": str(profile_path),
        "profiles_planned": selected_profiles,
        "profiles_results": [],
    }
    run_meta_path = out_dir / "run_meta.json"
    _write_json(run_meta_path, run_meta)

    retrieval_macro_rows: List[Dict[str, Any]] = []
    e2e_macro_rows: List[Dict[str, Any]] = []

    for profile_name in selected_profiles:
        profile_overrides = deepcopy(profiles_map.get(profile_name) or {})
        description = str(profile_overrides.pop("description", ""))

        profile_cfg = resolve_profile_config(
            base_config=base_config,
            defaults=defaults,
            profile_name=profile_name,
            profile_overrides=profile_overrides,
            fixed_k=args.fixed_k,
            fixed_top_n=args.fixed_top_n,
        )
        llm_retrieval_cfg = profile_cfg.setdefault("llm_retrieval", {})
        timeout_cfg_value = _safe_int(llm_retrieval_cfg.get("request_timeout_sec", 600), 600)
        llm_retrieval_cfg["request_timeout_sec"] = timeout_cfg_value if timeout_cfg_value > 0 else 600

        status_row = {
            "profile": profile_name,
            "description": description,
            "status": "running",
            "started_at": _utc_now(),
        }
        run_meta["profiles_results"].append(status_row)
        _write_json(run_meta_path, run_meta)

        try:
            result = run_profile(
                run_id=run_id,
                profile_name=profile_name,
                profile_cfg=profile_cfg,
                queries=queries,
                out_dir=out_dir,
            )
            status_row.update(
                {
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "records": result.get("records", 0),
                    "track": result.get("track", "retrieval"),
                    "backend_mode_effective": result.get("backend_mode_effective", "unknown"),
                    "backend_mode_config": result.get("backend_mode_config", "unknown"),
                    "backend_mode_required": result.get("backend_mode_required", "any"),
                    "config_hash": result.get("config_hash", ""),
                }
            )
            if result.get("track") == "e2e":
                e2e_macro_rows.extend(result.get("macro_micro_rows", []))
            else:
                retrieval_macro_rows.extend(result.get("macro_micro_rows", []))
        except Exception as exc:
            status_row.update(
                {
                    "status": "failed",
                    "completed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

        _write_json(run_meta_path, run_meta)

    _write_csv(out_dir / "retrieval_macro_micro.csv", retrieval_macro_rows)
    _write_csv(out_dir / "e2e_macro_micro.csv", e2e_macro_rows)

    run_meta["completed_at"] = _utc_now()
    run_meta["status"] = "completed"
    run_meta["counts"] = {
        "planned": len(selected_profiles),
        "completed": len([r for r in run_meta["profiles_results"] if r.get("status") == "completed"]),
        "failed": len([r for r in run_meta["profiles_results"] if r.get("status") == "failed"]),
    }
    if run_meta["counts"]["failed"] > 0:
        run_meta["status"] = "completed_with_issues"
    _write_json(run_meta_path, run_meta)

    print(f"[Ablation] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
