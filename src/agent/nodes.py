import threading
import json
import time
import re
from copy import deepcopy
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import src.utils
from src.agent.state import GraphState
from src.agent.chains import get_grade_chain, get_rewrite_chain, get_hallucination_chain
from src.llm.factory import get_llm
from src.llm.json_retry import parse_json_strict, repair_json_text
from src.llm.rag_generate import generate_answer
from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.retrieve.split_budget import compute_split_budgets

def _build_structured_metadata(d) -> Dict[str, str]:
    """
    Build structured metadata from source_file split.
    Fallback is used only when split fields are missing.
    """
    source_file = str(d.metadata.get("source_file", "unknown"))
    parts = [p for p in source_file.split("/") if p]

    language = d.metadata.get("lang")
    if not language:
        if len(parts) >= 2 and parts[0] == "doc":
            language = parts[1]
        else:
            language = "unknown"

    filename = parts[-1] if parts else "unknown"

    topic = parts[-2] if len(parts) >= 2 else ""
    if not topic:
        # Error-only fallback
        fallback_header = str(d.metadata.get("original_header", "")).strip()
        topic = fallback_header or "unknown"

    return {
        "Language": str(language),
        "Filename": str(filename),
        "Topic": str(topic),
        "SourceFile": source_file,
    }


def _format_structured_block(meta: Dict[str, str]) -> str:
    return (
        "[Structured Metadata]\n"
        f"Language: {meta['Language']}\n"
        f"Filename: {meta['Filename']}\n"
        f"Topic: {meta['Topic']}\n"
        f"SourceFile: {meta['SourceFile']}"
    )

def _get_generation_timeout_sec(default_sec: int = 600) -> int:
    try:
        config = src.utils.load_config()
        sec = int(config.get("rag", {}).get("generation_timeout_sec", default_sec))
        return sec if sec > 0 else default_sec
    except Exception:
        return default_sec


def _is_timeout_exception(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "timeout" in msg
        or "read timed out" in msg
        or "timed out" in msg
    )


def _append_warning(
    warnings: List[str],
    timeout_locations: List[str],
    message: str,
    timeout_location: str | None = None,
) -> None:
    if message not in warnings:
        warnings.append(message)
    if timeout_location and timeout_location not in timeout_locations:
        timeout_locations.append(timeout_location)


def _get_hallucination_settings() -> Dict[str, Any]:
    try:
        rag_cfg = src.utils.load_config().get("rag", {}) or {}
    except Exception:
        rag_cfg = {}
    mode = str(rag_cfg.get("hallucination_mode", "two_stage")).strip().lower()
    if mode not in {"legacy", "two_stage"}:
        mode = "two_stage"
    ambiguous_policy = str(
        rag_cfg.get("hallucination_ambiguous_policy", "pass_with_warning")
    ).strip().lower()
    if ambiguous_policy not in {"pass_with_warning", "rewrite"}:
        ambiguous_policy = "pass_with_warning"
    return {
        "hallucination_mode": mode,
        "hallucination_ambiguous_policy": ambiguous_policy,
        "skip_rewrite_on_stable_retrieval": bool(
            rag_cfg.get("skip_rewrite_on_stable_retrieval", True)
        ),
        "stable_retrieval_jaccard_threshold": float(
            rag_cfg.get("stable_retrieval_jaccard_threshold", 0.9)
        ),
    }


def _get_generation_settings() -> Dict[str, Any]:
    try:
        rag_cfg = src.utils.load_config().get("rag", {}) or {}
    except Exception:
        rag_cfg = {}
    return {
        "grounding_mode": str(rag_cfg.get("grounding_mode", "strict_extractive")).strip().lower(),
        "require_inline_citations": bool(rag_cfg.get("require_inline_citations", True)),
        "max_claims_per_answer": max(1, int(rag_cfg.get("max_claims_per_answer", 12))),
        "missing_info_phrase": str(rag_cfg.get("missing_info_phrase", "정보가 부족합니다")).strip() or "정보가 부족합니다",
        "grounding_min_citation_coverage": float(rag_cfg.get("grounding_min_citation_coverage", 1.0)),
        "fail_closed_on_guard": bool(rag_cfg.get("fail_closed_on_guard", False)),
    }


def _doc_fingerprint_key(d: Any) -> str:
    ref = _doc_canonical_ref(d).strip()
    if ref:
        return ref
    return "|".join([str(v) for v in _doc_identity(d)])


def _compute_retrieval_stability(history: List[List[str]], threshold: float) -> Dict[str, Any]:
    if len(history) < 2:
        return {
            "stable": False,
            "jaccard": 0.0,
            "threshold": threshold,
            "prev_count": 0,
            "curr_count": len(history[-1]) if history else 0,
        }
    prev = set(history[-2])
    curr = set(history[-1])
    union = prev | curr
    jaccard = (len(prev & curr) / len(union)) if union else 1.0
    return {
        "stable": bool(jaccard >= threshold and len(curr) > 0),
        "jaccard": jaccard,
        "threshold": threshold,
        "prev_count": len(prev),
        "curr_count": len(curr),
    }


def _canonical_ref_candidates(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\.html:\d+"
    refs = re.findall(pattern, text)
    return list(dict.fromkeys([r.strip() for r in refs if r and r.strip()]))


def _doc_canonical_ref(d: Any) -> str:
    ref_id = str(d.metadata.get("ref_id", "")).strip()
    if ref_id:
        canon = _canonical_ref_candidates(ref_id)
        if canon:
            return canon[0]
        return ref_id.split(" [", 1)[0].strip()

    source_file = str(d.metadata.get("source_file", "")).strip().replace("\\", "/").lstrip("/")
    chunk_id = d.metadata.get("chunk_id")
    if source_file and chunk_id is not None:
        parts = [p for p in source_file.split("/") if p]
        if len(parts) >= 3 and parts[0] == "doc":
            return f"{parts[1]}/{parts[-1]}:{chunk_id}"
        return f"{source_file}:{chunk_id}"
    return source_file


def _allowed_ref_tags(documents: List[Any]) -> set[str]:
    refs: set[str] = set()
    for d in documents:
        ref_id = str(d.metadata.get("ref_id", "")).strip()
        source_file = str(d.metadata.get("source_file", "")).strip()
        chunk_id = d.metadata.get("chunk_id")
        if ref_id:
            refs.add(ref_id)
            trimmed_ref = ref_id.split(" [", 1)[0].strip()
            if trimmed_ref:
                refs.add(trimmed_ref)
            for canonical_ref in _canonical_ref_candidates(ref_id):
                refs.add(canonical_ref)
        if source_file:
            refs.add(source_file)
            if chunk_id is not None:
                refs.add(f"{source_file}:{chunk_id}")
        canonical_from_doc = _doc_canonical_ref(d)
        if canonical_from_doc:
            refs.add(canonical_from_doc)
    return refs


def _extract_bracket_tags(text: str) -> List[str]:
    tags = re.findall(r"\[([^\[\]\n]{1,160})\]", text or "")
    return [t.strip() for t in tags if t and t.strip()]


def _extract_citation_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    seen: set[str] = set()

    def _push(token: str) -> None:
        normalized = (token or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tokens.append(normalized)

    for tag in _extract_bracket_tags(text):
        canonical = _canonical_ref_candidates(tag)
        if canonical:
            for c in canonical:
                _push(c)
            continue
        normalized = tag.split(" [", 1)[0].strip()
        _push(normalized)

    for canonical in _canonical_ref_candidates(text):
        _push(canonical)

    return tokens


def _is_claim_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.endswith(":") and len(s) <= 40:
        return False
    lower = s.lower()
    if lower.startswith("reference sources"):
        return False
    if s.startswith("> [!WARNING]"):
        return False
    if re.fullmatch(r"[-*#\d\.\)\(\s:]+", s):
        return False
    return any(ch.isalnum() for ch in s)


def _is_markdown_table_separator(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if "|" not in s:
        return False
    compact = s.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return compact == ""


def _parse_grade_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "no"
    token = re.split(r"\s+", text, maxsplit=1)[0].strip(".,;:()[]{}\"'")
    if token in {"yes", "no", "no_topic", "no_other"}:
        return "yes" if token == "yes" else "no"
    if "yes" in text and "no" not in text:
        return "yes"
    return "no"


def _compute_citation_coverage(generation: str, allowed_refs: set[str]) -> Dict[str, Any]:
    raw_lines = (generation or "").splitlines()
    claim_lines: List[str] = []
    for idx, raw in enumerate(raw_lines):
        ln = raw.strip()
        if not _is_claim_line(ln):
            continue
        if _is_markdown_table_separator(ln):
            continue
        if "|" in ln:
            j = idx + 1
            while j < len(raw_lines) and not raw_lines[j].strip():
                j += 1
            if j < len(raw_lines) and _is_markdown_table_separator(raw_lines[j].strip()):
                # Markdown table header row is not a factual claim.
                continue
        claim_lines.append(ln)
    tagged_claims = 0
    unknown_tags: List[str] = []
    for ln in claim_lines:
        tags = _extract_citation_tokens(ln)
        if not tags:
            continue
        has_allowed = any(tag in allowed_refs for tag in tags)
        if has_allowed:
            tagged_claims += 1
        for tag in tags:
            if tag not in allowed_refs and ("/" in tag or ".html" in tag):
                unknown_tags.append(tag)
    claim_count = len(claim_lines)
    coverage = (tagged_claims / claim_count) if claim_count > 0 else 0.0
    return {
        "claim_count": claim_count,
        "tagged_claim_count": tagged_claims,
        "inline_citation_coverage": coverage,
        "unknown_citation_tags": sorted(set(unknown_tags)),
    }


def _build_fail_closed_answer(documents: List[Any], missing_info_phrase: str, max_claims: int) -> str:
    lines: List[str] = ["근거 기반 제한 응답:"]
    seen: set[str] = set()
    added = 0
    for d in documents:
        ref_id = _doc_canonical_ref(d)
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        content = re.sub(r"\s+", " ", (d.page_content or "").strip()).replace("^", "").strip()
        header = str(d.metadata.get("original_header", "")).strip()
        if not content:
            continue
        summary = content[:120]
        if header:
            summary = f"{header}: {summary}"
        lines.append(f"- {summary} [{ref_id}]")
        added += 1
        if added >= max_claims:
            break
    if added == 0:
        lines.append(f"- {missing_info_phrase}.")
    else:
        lines.append(f"- 위 근거 외 세부사항은 {missing_info_phrase}.")
    return "\n".join(lines)


def _apply_grounding_guard(generation: str, documents: List[Any]) -> tuple[str, Dict[str, Any]]:
    settings = _get_generation_settings()
    mode = settings["grounding_mode"]
    require_inline = bool(settings["require_inline_citations"])
    min_cov = max(0.0, min(1.0, float(settings["grounding_min_citation_coverage"])))
    max_claims = int(settings["max_claims_per_answer"])
    missing_phrase = settings["missing_info_phrase"]
    fail_closed_on_guard = bool(settings["fail_closed_on_guard"])
    allowed_refs = _allowed_ref_tags(documents)
    cov_stats = _compute_citation_coverage(generation, allowed_refs)
    guard_triggered = False
    reason = "none"
    fail_closed_applied = False

    if mode == "strict_extractive":
        if cov_stats["claim_count"] > max_claims:
            guard_triggered = True
            reason = "claim_count_exceeded"
        elif require_inline and cov_stats["inline_citation_coverage"] < min_cov:
            guard_triggered = True
            reason = "citation_coverage_low"
        elif cov_stats["unknown_citation_tags"]:
            guard_triggered = True
            reason = "unknown_citation_tags"

    if guard_triggered and fail_closed_on_guard:
        generation = _build_fail_closed_answer(
            documents=documents,
            missing_info_phrase=missing_phrase,
            max_claims=max_claims,
        )
        cov_stats = _compute_citation_coverage(generation, allowed_refs)
        fail_closed_applied = True

    diagnostics = {
        "grounding_mode": mode,
        "require_inline_citations": require_inline,
        "max_claims_per_answer": max_claims,
        "fail_closed_on_guard": fail_closed_on_guard,
        "fail_closed_applied": fail_closed_applied,
        "inline_citation_coverage": cov_stats["inline_citation_coverage"],
        "claim_count": cov_stats["claim_count"],
        "tagged_claim_count": cov_stats["tagged_claim_count"],
        "unknown_citation_tags": cov_stats["unknown_citation_tags"],
        "unsupported_claim_guard_triggered": guard_triggered,
        "fail_closed_reason": reason,
    }
    return generation, diagnostics


def _generate_with_timeout(query: str, context: str, timeout_sec: int) -> str:
    result = {"value": None, "error": None}
    timeout_fallback = (
        f"정보가 부족합니다\n\n(답변 생성 시간 제한 {timeout_sec}초를 초과했습니다.)"
    )

    def _worker():
        try:
            result["value"] = generate_answer(
                query=query,
                context=context,
                request_timeout_sec=timeout_sec,
            )
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return timeout_fallback

    if result["error"] is not None:
        if "timeout" in str(result["error"]).lower():
            return timeout_fallback
        raise result["error"]

    return result["value"]


def _doc_identity(doc) -> tuple:
    source_file = doc.metadata.get("source_file")
    chunk_id = doc.metadata.get("chunk_id")
    if source_file is not None and chunk_id is not None:
        return (str(source_file), str(chunk_id))
    ref_id = doc.metadata.get("ref_id")
    if ref_id:
        return ("ref", str(ref_id))
    db_id = doc.metadata.get("_db_id")
    if db_id is not None:
        return ("db", str(db_id))
    return ("obj", str(id(doc)))


def _doc_score(doc) -> float:
    try:
        score = doc.metadata.get("score")
        if score is None:
            return float("-inf")
        return float(score)
    except Exception:
        return float("-inf")


def _generate_split_queries(
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

    def _parse_split_json(raw: str) -> Dict[str, Any]:
        parsed = parse_json_strict(raw)
        if not isinstance(parsed, dict):
            raise ValueError("split_query_invalid_schema")
        return parsed

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
                parsed = _parse_split_json(raw)
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
                parsed = _parse_split_json(repaired_raw)
            queries = parsed.get("queries", {}) if isinstance(parsed, dict) else {}
            if not isinstance(queries, dict):
                raise ValueError("split_query_invalid_schema")

            out: Dict[str, str] = {}
            for lang in languages:
                q = queries.get(lang)
                if not isinstance(q, str) or not q.strip():
                    raise ValueError(f"split_query_missing_lang:{lang}")
                out[lang] = q.strip()
            return out, json_retry_count
        except Exception as e:
            last_error = e
            if attempt >= attempts:
                break
            sleep_sec = max(0.0, backoff_sec) * attempt
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    raise RuntimeError(f"split_query_generation_failed: {last_error}")


def _merge_split_docs(
    branch_docs: Dict[str, List[Any]],
    language_order: List[str],
    merge_cap: int,
) -> List[Any]:
    sorted_docs: Dict[str, List[Any]] = {}
    for lang in language_order:
        docs = list(branch_docs.get(lang, []))
        docs.sort(key=lambda d: _doc_score(d), reverse=True)
        sorted_docs[lang] = docs

    merged: List[Any] = []
    seen = set()
    cursor = {lang: 0 for lang in language_order}
    cap = max(0, int(merge_cap))
    if cap == 0:
        return []

    while len(merged) < cap:
        progressed = False
        for lang in language_order:
            docs = sorted_docs.get(lang, [])
            idx = cursor.get(lang, 0)
            while idx < len(docs):
                doc = docs[idx]
                idx += 1
                key = _doc_identity(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
                progressed = True
                break
            cursor[lang] = idx
            if len(merged) >= cap:
                break
        if not progressed:
            break
    return merged


def retrieve_node(state: GraphState):
    """
    Retrieve documents (L1 + L0) based on the question.
    Features: Dynamic Top-K, Accumulative Retrieval.
    """
    print("---RETRIEVE---")
    question = state["question"]
    search_count = state.get("search_count", 0)
    existing_docs = state.get("documents", []) # Accumulative
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))

    # Unload LLM to free VRAM for embeddings
    try:
        from src.utils import LLMUtility
        LLMUtility.unload_model("retrieval")
    except Exception:
        pass

    config = src.utils.load_config()
    retrieval_cfg = config.get("retrieval", {})
    reranker_cfg = config.get("reranker", {})
    dynamic_scaling = bool(retrieval_cfg.get("dynamic_scaling", True))

    if dynamic_scaling:
        # Dynamic Top-K scaling (3-tier)
        # Base: 15 for normal queries
        # Mid-broad: 30 for topic-specific linguistics queries
        # Super-broad: 60 for exhaustive queries (all, compare, list)
        super_broad_keywords = [
            "모든", "목록", "전체", "list", "all", "overview", "언어들",
            "데이터베이스", "db", "비교", "compare", "차이", "difference"
        ]

        mid_broad_keywords = [
            # Topic-specific linguistics terms
            "aspect", "시상", "tense", "양태", "modality", "mood",
            "연동", "serial", "svc", "구문", "construction",
            "음운", "phonology", "phoneme", "자음", "모음", "vowel", "consonant",
            "어순", "word order", "화제", "topic", "주어", "subject",
            "격", "case", "표지", "marker", "접사", "affix",
            "종류", "types", "features", "특징", "구조", "structure"
        ]

        q_lower = question.lower()

        if any(kw in q_lower for kw in super_broad_keywords):
            k, top_n = 60, 25
            print("  - Dynamic Scaling: Super-broad query detected.")
            print(f"    -> k={k}, top_n={top_n}")
        elif any(kw in q_lower for kw in mid_broad_keywords):
            k, top_n = 30, 15
            print("  - Dynamic Scaling: Mid-broad (topic-specific) query detected.")
            print(f"    -> k={k}, top_n={top_n}")
        else:
            k, top_n = 15, 10
            print(f"  - Dynamic Scaling: Normal query. k={k}, top_n={top_n}")
    else:
        k = int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 15)))
        top_n = int(retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 10)))
        print(f"  - Dynamic Scaling: disabled. Using fixed k={k}, top_n={top_n}")


    with RAGRetriever() as retriever:
        # Metadata Extraction Logic (New)
        print("  - Analyzing Query Metadata...")
        metadata = extract_query_metadata(question, retriever.config)
        retrieval_diagnostics = metadata.get("_diagnostics", {})
        if not isinstance(retrieval_diagnostics, dict):
            retrieval_diagnostics = {}
            metadata["_diagnostics"] = retrieval_diagnostics
        retrieval_diagnostics.setdefault("split_applied", False)
        retrieval_diagnostics.setdefault("split_branches", [])
        retrieval_diagnostics.setdefault("split_json_retry_count", 0)
        retrieval_diagnostics.setdefault("split_json_fail_open", False)
        timeout_loc = retrieval_diagnostics.get("timeout_location", "none")
        if timeout_loc != "none":
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_loc}")
            _append_warning(
                warnings,
                timeout_locations,
                f"Language detection timeout at {timeout_loc}; proceeding without language filter.",
                timeout_loc,
            )

        query_split_cfg = retrieval_cfg.get("query_split", {})
        split_enabled = bool(query_split_cfg.get("enabled", True))
        max_languages = int(query_split_cfg.get("max_languages", 3))
        branch_mode = str(query_split_cfg.get("branch_k_mode", "full_per_branch"))
        llm_cfg = config.get("llm_retrieval", {})
        split_retries = int(llm_cfg.get("lang_detect_retries", 2))
        split_backoff_sec = float(llm_cfg.get("lang_detect_backoff_sec", 0.5))
        split_json_retries = int(llm_cfg.get("lang_json_retries", 1))
        split_json_backoff_sec = float(llm_cfg.get("lang_json_backoff_sec", 0.25))

        detected_languages = []
        if isinstance(metadata.get("lang"), list):
            detected_languages = [str(v).strip() for v in metadata.get("lang", []) if str(v).strip()]
        if not detected_languages:
            detected_languages = [
                str(v).strip()
                for v in retrieval_diagnostics.get("detected_languages", [])
                if str(v).strip()
            ]

        if (
            split_enabled
            and branch_mode in {"balanced", "full_per_branch"}
            and len(detected_languages) >= 2
        ):
            languages = detected_languages[:max(2, max_languages)]
            try:
                split_queries, split_json_retry_count = _generate_split_queries(
                    question=question,
                    languages=languages,
                    retries=max(0, split_retries),
                    backoff_sec=max(0.0, split_backoff_sec),
                    json_retries=max(0, split_json_retries),
                    json_backoff_sec=max(0.0, split_json_backoff_sec),
                )
                budgets = compute_split_budgets(
                    total_k=k,
                    total_top_n=top_n,
                    branch_count=len(languages),
                    mode=branch_mode,
                )
                split_branch_docs: Dict[str, List[Any]] = {}

                for i, lang in enumerate(languages):
                    branch_k = budgets.k_allocs[i]
                    if branch_k <= 0:
                        continue
                    branch_top_n = max(1, budgets.top_n_allocs[i])
                    branch_metadata = deepcopy(metadata)
                    branch_metadata["lang"] = lang
                    branch_diag = branch_metadata.setdefault("_diagnostics", {})
                    branch_diag["split_applied"] = True
                    branch_diag["split_branches"] = languages
                    branch_docs = retriever.retrieve_documents(
                        split_queries[lang],
                        k=branch_k,
                        top_n=branch_top_n,
                        metadata=branch_metadata,
                    )
                    split_branch_docs[lang] = branch_docs

                new_docs = _merge_split_docs(
                    branch_docs=split_branch_docs,
                    language_order=languages,
                    merge_cap=max(1, budgets.merge_cap),
                )
                retrieval_diagnostics["split_applied"] = True
                retrieval_diagnostics["split_branches"] = languages
                retrieval_diagnostics["detected_languages"] = languages
                retrieval_diagnostics["filter_applied"] = True
                retrieval_diagnostics["split_json_retry_count"] = int(split_json_retry_count)
                retrieval_diagnostics["split_json_fail_open"] = False
                print(
                    "SPLIT_QUERY_DIAGNOSTICS "
                    f"split_applied=true split_branches={languages} "
                    f"branch_mode={branch_mode} merge_cap={max(1, budgets.merge_cap)} "
                    f"k_allocs={budgets.k_allocs} top_n_allocs={budgets.top_n_allocs} "
                    f"split_json_retry_count={split_json_retry_count}"
                )
            except Exception as e:
                retrieval_diagnostics["split_applied"] = False
                retrieval_diagnostics["split_branches"] = []
                retrieval_diagnostics["split_json_fail_open"] = True
                print(f"SPLIT_QUERY_FALLBACK reason={e}")
                _append_warning(
                    warnings,
                    timeout_locations,
                    f"Split-query generation/retrieval failed ({e}); fallback to one-shot multi-language retrieval.",
                    None,
                )
                if _is_timeout_exception(e):
                    timeout_loc = "split_query_generation"
                    print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_loc}")
                    _append_warning(
                        warnings,
                        timeout_locations,
                        f"Split-query timeout at {timeout_loc}; fallback applied.",
                        timeout_loc,
                    )
                new_docs = retriever.retrieve_documents(question, k=k, top_n=top_n, metadata=metadata)
        else:
            new_docs = retriever.retrieve_documents(question, k=k, top_n=top_n, metadata=metadata)
    
    # Merge and deduplicate
    combined_docs = []
    seen_ids = set()
    for d in existing_docs + new_docs:
        ref_id = d.metadata.get('ref_id')
        if ref_id and ref_id not in seen_ids:
            combined_docs.append(d)
            seen_ids.add(ref_id)
        elif not ref_id: # fallback
             combined_docs.append(d)

    print(f"  - Total Documents (Accumulated): {len(combined_docs)}")
    current_fingerprint_set = set()
    for d in combined_docs:
        key = _doc_fingerprint_key(d)
        if key:
            current_fingerprint_set.add(key)
    current_fingerprint = sorted(current_fingerprint_set)
    retrieval_fingerprints = (retrieval_fingerprints + [current_fingerprint])[-8:]
    retrieval_diagnostics["retrieval_fingerprint_size"] = len(current_fingerprint)
    
    return {
        "documents": combined_docs,
        "question": question,
        "search_count": search_count,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
    }

def grade_documents_node(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))

    # Skip grading if disabled via config
    config = src.utils.load_config()
    if config.get("rag", {}).get("skip_grading", False):
        print("---SKIP DOCUMENT GRADING (disabled)---")
        retrieval_diagnostics["grading_topic_gate_applied"] = False
        return {
            "documents": documents,
            "question": question,
            "retrieval_diagnostics": retrieval_diagnostics,
            "retrieval_fingerprints": retrieval_fingerprints,
            "warnings": warnings,
            "timeout_locations": timeout_locations,
        }

    retrieval_diagnostics["grading_topic_gate_applied"] = False
    retrieval_diagnostics["grading_rejected_by_topic"] = 0
    retrieval_diagnostics["grading_rejected_other"] = 0
    retrieval_diagnostics["grading_relevant_count"] = 0

    try:
        grader = get_grade_chain()
    except Exception as e:
        timeout_location = "grade_documents"
        print(f"  - Grader init failed. Fail-open mode: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Document grading failed at {timeout_location}; all documents retained.",
            timeout_location,
        )
        return {
            "documents": documents,
            "question": question,
            "retrieval_diagnostics": retrieval_diagnostics,
            "retrieval_fingerprints": retrieval_fingerprints,
            "warnings": warnings,
            "timeout_locations": timeout_locations,
        }

    filtered_docs = []
    
    relevance_count = 0
    
    for d in documents:
        structured_meta = _build_structured_metadata(d)
        source_info = f"[Source]: {structured_meta['SourceFile']}"
        rich_content = (
            f"{_format_structured_block(structured_meta)}\n"
            "[Document Content]\n"
            f"{d.page_content}"
        )
        
        try:
            score = grader.invoke({"question": question, "document": rich_content})
        except Exception as e:
            timeout_location = "grade_documents"
            print(f"  - Grading failed for {d.metadata.get('ref_id')}. Fail-open keep doc: {e}")
            if _is_timeout_exception(e):
                print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
            _append_warning(
                warnings,
                timeout_locations,
                f"Document grading timeout/error at {timeout_location}; fail-open applied.",
                timeout_location,
            )
            filtered_docs.append(d)
            continue
        
        # Robust parsing for label output
        grade_label = _parse_grade_label(score)

        if grade_label == "yes":
            print(f"  - GRADED DOCUMENT: RELEVANT ({d.metadata.get('ref_id')}) [{source_info}]")
            filtered_docs.append(d)
            relevance_count += 1
            retrieval_diagnostics["grading_relevant_count"] = (
                int(retrieval_diagnostics.get("grading_relevant_count", 0)) + 1
            )
        else:
            print(f"  - GRADED DOCUMENT: NOT RELEVANT ({d.metadata.get('ref_id')})")
            retrieval_diagnostics["grading_rejected_other"] = (
                int(retrieval_diagnostics.get("grading_rejected_other", 0)) + 1
            )
            continue

    if relevance_count == 0 and documents:
        # Model-specific grader failures can mark every doc as "no".
        # Fail-open with a small head slice to avoid rewrite-only loops.
        fallback_keep = min(3, len(documents))
        filtered_docs = list(documents[:fallback_keep])
        retrieval_diagnostics["grading_all_no_fail_open"] = True
        retrieval_diagnostics["grading_all_no_fallback_kept"] = fallback_keep
        print(
            f"  - GRADING FAIL-OPEN: all docs graded non-relevant; "
            f"keeping top {fallback_keep} docs for downstream generation."
        )
    else:
        retrieval_diagnostics["grading_all_no_fail_open"] = False
        retrieval_diagnostics["grading_all_no_fallback_kept"] = 0
            
    return {
        "documents": filtered_docs,
        "question": question,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
    }

def generate_node(state: GraphState):
    """
    Generate answer using the retrieved documents.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))
    
    # Format Context
    text_context_lines = []
    vision_context_lines = []
    
    for d in documents:
        if d.metadata.get('type') == 'vision':
            vision_context_lines.append(d.page_content)
        else:
            structured_meta = _build_structured_metadata(d)
            level = d.metadata.get('level', '0')
            prefix = "[Summary]" if str(level) == '1' else "[Detail]"
            source_info = f"[Source: {structured_meta['SourceFile']}]"
            text_context_lines.append(
                f"{prefix} Ref: {d.metadata.get('ref_id', 'Chunk')} {source_info}\n"
                f"{_format_structured_block(structured_meta)}\n"
                "[Document Content]\n"
                f"{d.page_content}"
            )

    context_str = "\n\n".join(text_context_lines)
    if vision_context_lines:
        context_str += "\n\n[Visual Evidence Found]\n" + "\n---\n".join(vision_context_lines)
        
    if not context_str.strip():
        context_str = "No relevant documents found."

    timeout_sec = _get_generation_timeout_sec(default_sec=600)
    generation = _generate_with_timeout(query=question, context=context_str, timeout_sec=timeout_sec)
    generation, generation_diagnostics = _apply_grounding_guard(generation=generation, documents=documents)
    if generation_diagnostics.get("unsupported_claim_guard_triggered", False):
        guard_reason = generation_diagnostics.get("fail_closed_reason", "unknown")
        fail_closed_applied = bool(generation_diagnostics.get("fail_closed_applied", False))
        mode_tag = "fail-closed" if fail_closed_applied else "fail-open-warning"
        warning_msg = f"Generation grounding guard triggered ({mode_tag}, reason={guard_reason})."
        if warning_msg not in warnings:
            warnings.append(warning_msg)
        print(
            "GENERATION_GUARD_DIAGNOSTICS "
            f"triggered=true reason={guard_reason} "
            f"fail_closed_applied={str(fail_closed_applied).lower()} "
            f"coverage={generation_diagnostics.get('inline_citation_coverage', 0.0):.3f} "
            f"claim_count={generation_diagnostics.get('claim_count', 0)}"
        )
    else:
        print(
            "GENERATION_GUARD_DIAGNOSTICS "
            f"triggered=false coverage={generation_diagnostics.get('inline_citation_coverage', 0.0):.3f} "
            f"claim_count={generation_diagnostics.get('claim_count', 0)}"
        )
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_diagnostics": generation_diagnostics,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
    }

def check_hallucination_node(state: GraphState):
    """
    Checks if the generated answer is grounded in the documents.
    """
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    generation_diagnostics = dict(state.get("generation_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))
    hallu_settings = _get_hallucination_settings()
    generation_settings = _get_generation_settings()
    min_cov = max(
        0.0,
        min(1.0, float(generation_settings.get("grounding_min_citation_coverage", 1.0))),
    )
    try:
        max_claims = int(
            generation_diagnostics.get(
                "max_claims_per_answer",
                generation_settings.get("max_claims_per_answer", 12),
            )
        )
    except Exception:
        max_claims = int(generation_settings.get("max_claims_per_answer", 12))
    stable_threshold = max(
        0.0,
        min(1.0, float(hallu_settings.get("stable_retrieval_jaccard_threshold", 0.9))),
    )
    stable_info = _compute_retrieval_stability(retrieval_fingerprints, stable_threshold)
    generation_diagnostics["hallucination_mode"] = hallu_settings["hallucination_mode"]
    generation_diagnostics["retrieval_stable"] = stable_info["stable"]
    generation_diagnostics["retrieval_jaccard"] = stable_info["jaccard"]
    generation_diagnostics["rewrite_skipped_due_to_stability"] = False
    generation_diagnostics["stage1_failed_metric"] = "none"

    print(
        "RETRIEVAL_STABILITY_DIAGNOSTICS "
        f"stable={str(stable_info['stable']).lower()} "
        f"jaccard={stable_info['jaccard']:.3f} "
        f"threshold={stable_info['threshold']:.3f} "
        f"prev_count={stable_info['prev_count']} curr_count={stable_info['curr_count']}"
    )

    def _pack(status: str, text: str) -> Dict[str, Any]:
        return {
            "documents": documents,
            "generation": text,
            "hallucination_status": status,
            "retrieval_diagnostics": retrieval_diagnostics,
            "generation_diagnostics": generation_diagnostics,
            "warnings": warnings,
            "timeout_locations": timeout_locations,
            "retrieval_fingerprints": retrieval_fingerprints,
        }

    # Skip hallucination check if disabled via config
    config = src.utils.load_config()
    if config.get("rag", {}).get("skip_hallucination", False):
        print("---SKIP HALLUCINATION CHECK (disabled)---")
        return _pack("pass", generation)

    try:
        checker = get_hallucination_chain()
    except Exception as e:
        timeout_location = "hallucination_check"
        print(f"  - Hallucination checker init failed. Fail-open pass-with-warning: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Hallucination checker init timeout/error at {timeout_location}; fail-open applied.",
            timeout_location,
        )
        warning_msg = (
            "\n\n> [!WARNING] (Runtime): Hallucination check initialization timed out/failed. "
            "Answer is returned in fail-open mode."
        )
        return _pack("pass", generation + warning_msg)
    
    # Prepare documents text for checker
    full_text = "\n".join([d.page_content for d in documents])
    
    if len(full_text) > 30000:
        docs_text = full_text[:30000] + "... (truncated)"
    else:
        docs_text = full_text
        
    # Check for valid refusal
    refusal_keywords = ["정보가 부족", "알 수 없습니다", "문맥에 나타나 있지 않으", "제공된 문맥", "information is missing"]
    if any(k in generation for k in refusal_keywords):
        print("  - DECISION: GROUNDED REFUSAL (Pass)")
        return _pack("pass", generation)

    if hallu_settings["hallucination_mode"] == "two_stage":
        try:
            coverage = float(generation_diagnostics.get("inline_citation_coverage", 0.0))
        except Exception:
            coverage = 0.0
        try:
            claim_count = int(generation_diagnostics.get("claim_count", 0))
        except Exception:
            claim_count = 0
        unknown_tags = list(generation_diagnostics.get("unknown_citation_tags", []) or [])
        stage1_pass = bool(
            coverage >= min_cov
            and claim_count <= max_claims
            and not unknown_tags
        )
        stage1_failed_metric = "none"
        if coverage < min_cov:
            stage1_failed_metric = "coverage"
        elif claim_count > max_claims:
            stage1_failed_metric = "claim_count"
        elif unknown_tags:
            stage1_failed_metric = "unknown_tags"
        generation_diagnostics["stage1_failed_metric"] = stage1_failed_metric
        print(
            "HALLUCINATION_STAGE1_DIAGNOSTICS "
            f"pass={str(stage1_pass).lower()} "
            f"coverage={coverage:.3f} min_cov={min_cov:.3f} "
            f"claim_count={claim_count} max_claims={max_claims} "
            f"unknown_tags={len(unknown_tags)}"
        )
        if not stage1_pass:
            reason = "stage1_guard_failed"
            generation_diagnostics["hallucination_fail_reason"] = reason
            if (
                hallu_settings["skip_rewrite_on_stable_retrieval"]
                and stable_info["stable"]
            ):
                print("  - DECISION: STABLE RETRIEVAL + STAGE1 FAIL -> PASS WITH WARNING")
                warning_msg = (
                    "\n\n> [!WARNING] (Stability): 근거 규칙 검증에 경고가 있으나, "
                    "재검색 결과가 반복되어 현재 답변을 종료합니다."
                )
                _append_warning(
                    warnings,
                    timeout_locations,
                    "Hallucination stage-1 failed with stable retrieval; rewrite skipped.",
                    None,
                )
                generation_diagnostics["rewrite_skipped_due_to_stability"] = True
                return _pack("pass", generation + warning_msg)
            print("  - DECISION: STAGE1 GUARD FAIL -> REWRITE")
            return _pack("fail", generation)

    try:
        score = checker.invoke({"documents": docs_text, "generation": generation})
    except Exception as e:
        timeout_location = "hallucination_check"
        print(f"  - Hallucination check failed. Fail-open pass-with-warning: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Hallucination check timeout/error at {timeout_location}; fail-open applied.",
            timeout_location,
        )
        warning_msg = (
            "\n\n> [!WARNING] (Runtime): Hallucination check timed out/failed. "
            "Answer is returned in fail-open mode."
        )
        return _pack("pass", generation + warning_msg)

    grade = str(score).lower().strip()
    
    # Tiered Logic
    if "yes" in grade:
        print("  - DECISION: GROUNDED (Pass)")
        return _pack("pass", generation)
        
    elif "ambiguous" in grade:
        if hallu_settings["hallucination_ambiguous_policy"] == "pass_with_warning":
            print("  - DECISION: AMBIGUOUS -> PASS WITH WARNING")
            warning_msg = (
                "\n\n> [!WARNING] (Uncertainty): This answer is based on inference or partial information. "
                "Please verify specific details."
            )
            new_generation = generation + warning_msg
            return _pack("pass", new_generation)
        print("  - DECISION: AMBIGUOUS -> REWRITE")
        return _pack("fail", generation)
        
    else:
        generation_diagnostics["hallucination_fail_reason"] = "checker_no"
        if (
            hallu_settings["skip_rewrite_on_stable_retrieval"]
            and stable_info["stable"]
        ):
            print("  - DECISION: CHECKER NO + STABLE RETRIEVAL -> PASS WITH WARNING")
            warning_msg = (
                "\n\n> [!WARNING] (Stability): Hallucination checker가 보수적으로 실패를 반환했지만, "
                "재검색 결과가 반복되어 현재 답변을 종료합니다."
            )
            _append_warning(
                warnings,
                timeout_locations,
                "Hallucination checker returned no with stable retrieval; rewrite skipped.",
                None,
            )
            generation_diagnostics["rewrite_skipped_due_to_stability"] = True
            return _pack("pass", generation + warning_msg)
        print("  - DECISION: HALLUCINATION DETECTED (Fail)")
        return _pack("fail", generation)


def rewrite_query_node(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics", {}) or {})
    retrieval_fingerprints = [
        list(v) for v in state.get("retrieval_fingerprints", []) if isinstance(v, list)
    ]
    search_count = state.get("search_count", 0)
    warnings = list(state.get("warnings", []))
    timeout_locations = list(state.get("timeout_locations", []))

    better_question = question
    try:
        rewriter = get_rewrite_chain()
        better_question = rewriter.invoke({"question": question})
    except Exception as e:
        timeout_location = "rewrite_query"
        print(f"  - Rewrite failed. Fail-open keep original query: {e}")
        if _is_timeout_exception(e):
            print(f"TIMEOUT_DIAGNOSTIC timeout_location={timeout_location}")
        _append_warning(
            warnings,
            timeout_locations,
            f"Rewrite timeout/error at {timeout_location}; original query retained.",
            timeout_location,
        )
    
    print(f"  - Old Query: {question}")
    print(f"  - New Query: {better_question}")
    
    return {
        "documents": documents,
        "question": better_question,
        "search_count": search_count + 1,
        "retrieval_diagnostics": retrieval_diagnostics,
        "retrieval_fingerprints": retrieval_fingerprints,
        "warnings": warnings,
        "timeout_locations": timeout_locations,
    }
