#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieve.rag_retrieve import RAGRetriever, extract_query_metadata
from src.utils import load_config, use_config_override


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_profile_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults", {}) or {}
    profiles = data.get("profiles", {}) or {}
    if not profiles:
        raise ValueError(f"No profiles found in {path}")
    return {"defaults": defaults, "profiles": profiles}


def _resolve_profile_config(
    base_config: Dict[str, Any],
    defaults: Dict[str, Any],
    profile_overrides: Dict[str, Any],
    fixed_k: int | None,
    fixed_top_n: int | None,
) -> Dict[str, Any]:
    resolved = _deep_merge(base_config, defaults)
    resolved = _deep_merge(resolved, profile_overrides)

    retrieval = resolved.setdefault("retrieval", {})
    reranker = resolved.setdefault("reranker", {})
    ablation = resolved.setdefault("ablation", {})

    if ablation.get("force_deterministic", True):
        retrieval["dynamic_scaling"] = False
        if "fixed_k" not in retrieval:
            retrieval["fixed_k"] = retrieval.get("k", 15)
        if "fixed_top_n" not in retrieval:
            retrieval["fixed_top_n"] = reranker.get("top_n", 10)

    if fixed_k is not None:
        retrieval["fixed_k"] = fixed_k
    if fixed_top_n is not None:
        retrieval["fixed_top_n"] = fixed_top_n

    return resolved


def _load_queries(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    queries: List[Dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    qid = f"Q{i:03d}"
                    queries.append({"query_id": qid, "query_text": obj})
                    continue

                query_text = (
                    obj.get("query")
                    or obj.get("question")
                    or obj.get("text")
                )
                if not query_text:
                    continue
                query_id = (
                    obj.get("query_id")
                    or obj.get("id")
                    or obj.get("qid")
                    or f"Q{i:03d}"
                )
                queries.append({"query_id": str(query_id), "query_text": str(query_text)})
        return queries

    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            queries.append({"query_id": f"Q{i:03d}", "query_text": line})
    return queries


def _doc_key(meta: Dict[str, Any], rank: int) -> str:
    source_file = meta.get("source_file")
    chunk_id = meta.get("chunk_id")
    if source_file is not None and chunk_id is not None:
        return f"{source_file}:{chunk_id}"
    ref_id = meta.get("ref_id")
    if ref_id:
        return str(ref_id)
    db_id = meta.get("_db_id")
    if db_id is not None:
        return f"db:{db_id}"
    return f"rank:{rank}"


def _enforce_backend_mode(backend_mode: str, required_backend_mode: str) -> None:
    if required_backend_mode == "any":
        return
    if backend_mode != required_backend_mode:
        raise RuntimeError(
            f"Backend mode mismatch: required='{required_backend_mode}', actual='{backend_mode}'."
        )


def _enforce_reranker_policy(
    policy: str,
    expected_enabled: bool,
    actual_enabled: bool,
) -> None:
    if policy == "force_on":
        if not actual_enabled:
            raise RuntimeError("reranker_policy=force_on but reranker is not active.")
        return
    if policy == "force_off":
        if actual_enabled:
            raise RuntimeError("reranker_policy=force_off but reranker is active.")
        return
    if policy == "follow_profile" and actual_enabled != expected_enabled:
        raise RuntimeError(
            "reranker_policy=follow_profile mismatch: "
            f"config says {expected_enabled}, runtime is {actual_enabled}."
        )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_meta(path: Path, run_meta: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)


def run_profile(
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, str]],
    out_dir: Path,
) -> Dict[str, Any]:
    profile_dir = out_dir / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    retrieval_cfg = profile_cfg.get("retrieval", {})
    reranker_cfg = profile_cfg.get("reranker", {})
    ablation_cfg = profile_cfg.get("ablation", {})

    fixed_k = _safe_int(retrieval_cfg.get("fixed_k", retrieval_cfg.get("k", 15)), 15)
    fixed_top_n = _safe_int(
        retrieval_cfg.get("fixed_top_n", reranker_cfg.get("top_n", 10)),
        10,
    )
    required_backend_mode = str(ablation_cfg.get("required_backend_mode", "any")).lower()
    backend_mode_config = str((profile_cfg.get("database", {}) or {}).get("type", "postgres")).lower()
    reranker_policy = str(ablation_cfg.get("reranker_policy", "follow_profile")).lower()

    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    with use_config_override(profile_cfg):
        with RAGRetriever() as retriever:
            backend_mode_effective = str(getattr(retriever, "backend_mode", "unknown"))
            _enforce_backend_mode(backend_mode_effective, required_backend_mode)
            _enforce_reranker_policy(
                policy=reranker_policy,
                expected_enabled=bool(reranker_cfg.get("enabled", True)),
                actual_enabled=bool(retriever.use_reranker),
            )

            for item in queries:
                query_id = item["query_id"]
                query_text = item["query_text"]
                t_start = time.perf_counter()

                t_meta = time.perf_counter()
                metadata = extract_query_metadata(query_text, profile_cfg)
                metadata_elapsed_sec = time.perf_counter() - t_meta

                t_retrieve = time.perf_counter()
                docs = retriever.retrieve_documents(
                    query_text,
                    metadata=metadata,
                    k=fixed_k,
                    top_n=fixed_top_n,
                )
                retrieval_elapsed_sec = time.perf_counter() - t_retrieve
                total_elapsed_sec = time.perf_counter() - t_start

                diagnostics = metadata.get("_diagnostics", {}) if isinstance(metadata, dict) else {}
                timeout_location = diagnostics.get("timeout_location", "none")
                timeout_locations = [timeout_location] if timeout_location and timeout_location != "none" else []
                warnings = []
                if timeout_locations:
                    warnings.append(
                        f"Language detection timeout at {timeout_location}; retrieval continued fail-open."
                    )
                documents = []
                lang_counter: Counter = Counter()
                level_counter: Counter = Counter()

                for rank, doc in enumerate(docs, start=1):
                    meta = deepcopy(doc.metadata or {})
                    lang = meta.get("lang")
                    level = meta.get("level")
                    if lang is not None:
                        lang_counter[str(lang)] += 1
                    if level is not None:
                        level_counter[str(level)] += 1

                    documents.append(
                        {
                            "rank": rank,
                            "doc_key": _doc_key(meta, rank),
                            "db_id": meta.get("_db_id"),
                            "source_file": meta.get("source_file"),
                            "chunk_id": meta.get("chunk_id"),
                            "level": meta.get("level"),
                            "lang": lang,
                            "ref_id": meta.get("ref_id"),
                            "score": meta.get("score"),
                            "vector_rank": meta.get("_vector_rank"),
                        }
                    )

                resolved_flags = {
                    "enable_language_filter": bool(retrieval_cfg.get("enable_language_filter", True)),
                    "viking_use_detected_language": bool(
                        retrieval_cfg.get("viking_use_detected_language", True)
                    ),
                    "l0_only": bool(retrieval_cfg.get("l0_only", False)),
                    "hybrid_enabled": bool((retrieval_cfg.get("hybrid_search", {}) or {}).get("enabled", False)),
                    "recursive_retrieval": bool(retrieval_cfg.get("recursive_retrieval", True)),
                    "viking_enabled": bool((retrieval_cfg.get("viking", {}) or {}).get("enabled", False)),
                    "viking_mode": str((retrieval_cfg.get("viking", {}) or {}).get("mode", "soft")),
                    "reranker_enabled_config": bool(reranker_cfg.get("enabled", True)),
                    "reranker_applied": bool(retriever.use_reranker),
                }

                trace_row = {
                    "run_id": out_dir.name,
                    "profile": profile_name,
                    "query_id": query_id,
                    "query_text": query_text,
                    "timestamp": _utc_now(),
                    "backend_mode": backend_mode_effective,  # backward compatibility
                    "backend_mode_effective": backend_mode_effective,
                    "backend_mode_required": required_backend_mode,
                    "backend_mode_config": backend_mode_config,
                    "backend_enforcement_passed": True,
                    "config_hash": _config_hash(profile_cfg),
                    "k": fixed_k,
                    "top_n": fixed_top_n,
                    "metadata_elapsed_sec": round(metadata_elapsed_sec, 6),
                    "retrieval_elapsed_sec": round(retrieval_elapsed_sec, 6),
                    "total_elapsed_sec": round(total_elapsed_sec, 6),
                    "resolved_flags": resolved_flags,
                    "retrieval_diagnostics": diagnostics,
                    "documents": documents,
                    "doc_count": len(documents),
                    "doc_lang_counts": dict(lang_counter),
                    "doc_level_counts": dict(level_counter),
                    "warnings": warnings,
                    "timeout_locations": timeout_locations,
                }
                trace_rows.append(trace_row)

                summary_rows.append(
                    {
                        "run_id": trace_row["run_id"],
                        "profile": profile_name,
                        "query_id": query_id,
                        "backend_mode": backend_mode_effective,  # backward compatibility
                        "backend_mode_effective": backend_mode_effective,
                        "backend_mode_required": required_backend_mode,
                        "backend_mode_config": backend_mode_config,
                        "backend_enforcement_passed": True,
                        "k": fixed_k,
                        "top_n": fixed_top_n,
                        "doc_count": len(documents),
                        "l0_count": int(level_counter.get("0", 0)),
                        "l1_count": int(level_counter.get("1", 0)),
                        "metadata_elapsed_sec": round(metadata_elapsed_sec, 6),
                        "retrieval_elapsed_sec": round(retrieval_elapsed_sec, 6),
                        "total_elapsed_sec": round(total_elapsed_sec, 6),
                        "lang_detect_method": diagnostics.get("lang_detect_method", "none"),
                        "detected_languages": json.dumps(
                            diagnostics.get("detected_languages", []),
                            ensure_ascii=False,
                        ),
                        "detected_languages_candidates": json.dumps(
                            diagnostics.get("detected_languages_candidates", []),
                            ensure_ascii=False,
                        ),
                        "filter_applied": diagnostics.get("filter_applied", False),
                        "enable_language_filter": diagnostics.get(
                            "enable_language_filter",
                            resolved_flags["enable_language_filter"],
                        ),
                        "viking_use_detected_language": diagnostics.get(
                            "viking_use_detected_language",
                            resolved_flags["viking_use_detected_language"],
                        ),
                        "viking_language_seeded": diagnostics.get("viking_language_seeded", False),
                        "l0_only": diagnostics.get("l0_only", resolved_flags["l0_only"]),
                        "timeout_location": diagnostics.get("timeout_location", "none"),
                        "config_hash": trace_row["config_hash"],
                    }
                )

    _write_jsonl(profile_dir / "retrieval_traces.jsonl", trace_rows)
    _write_csv(profile_dir / "summary.csv", summary_rows)
    with (profile_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(profile_cfg, f, ensure_ascii=False, indent=2)

    return {
        "profile": profile_name,
        "records": len(trace_rows),
        "backend_mode": (trace_rows[0]["backend_mode"] if trace_rows else "unknown"),
        "backend_mode_effective": (
            trace_rows[0]["backend_mode_effective"] if trace_rows else "unknown"
        ),
        "backend_mode_required": required_backend_mode,
        "backend_mode_config": backend_mode_config,
        "backend_enforcement_passed": True,
        "config_hash": (trace_rows[0]["config_hash"] if trace_rows else _config_hash(profile_cfg)),
    }


def _run_profile_worker(
    queue: "mp.Queue",
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, str]],
    out_dir: str,
) -> None:
    try:
        summary = run_profile(
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            queries=queries,
            out_dir=Path(out_dir),
        )
        queue.put({"status": "completed", "summary": summary})
    except Exception as exc:
        queue.put(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_profile_with_timeout(
    profile_name: str,
    profile_cfg: Dict[str, Any],
    queries: List[Dict[str, str]],
    out_dir: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    if timeout_sec <= 0:
        summary = run_profile(
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            queries=queries,
            out_dir=out_dir,
        )
        return {"status": "completed", "summary": summary}

    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_run_profile_worker,
        args=(queue, profile_name, profile_cfg, queries, str(out_dir)),
        daemon=False,
    )

    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        return {
            "status": "timed_out",
            "error": f"Profile timed out after {timeout_sec}s",
            "timeout_sec": timeout_sec,
        }

    result: Dict[str, Any] | None = None
    if not queue.empty():
        result = queue.get()

    try:
        queue.close()
        queue.join_thread()
    except Exception:
        pass

    if result:
        return result

    if proc.exitcode == 0:
        return {
            "status": "failed",
            "error": "Profile process exited without result payload.",
        }

    return {
        "status": "failed",
        "error": f"Profile process exited with code {proc.exitcode}.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only E0-E5 ablation profiles.")
    parser.add_argument("--queries", required=True, help="Path to query file (.jsonl or .txt).")
    parser.add_argument(
        "--profile-config",
        default="config/ablation_profiles.yaml",
        help="Path to ablation profile config YAML.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names (default: all profiles in config).",
    )
    parser.add_argument(
        "--outdir",
        default="data/ablation_runs",
        help="Output base directory.",
    )
    parser.add_argument("--run-id", default="", help="Optional run id. Default: UTC timestamp.")
    parser.add_argument("--fixed-k", type=int, default=None, help="Override fixed k for all profiles.")
    parser.add_argument(
        "--fixed-top-n",
        type=int,
        default=None,
        help="Override fixed top_n for all profiles.",
    )
    parser.add_argument(
        "--profile-timeout-sec",
        type=int,
        default=None,
        help="Optional override timeout per profile (seconds). 0 disables timeout.",
    )
    args = parser.parse_args()

    base_config = load_config()
    profile_file = Path(args.profile_config)
    profile_data = _load_profile_file(profile_file)
    defaults = profile_data["defaults"]
    profiles_map = profile_data["profiles"]

    selected_profiles: List[str]
    if args.profiles.strip():
        selected_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    else:
        selected_profiles = list(profiles_map.keys())

    missing = [p for p in selected_profiles if p not in profiles_map]
    if missing:
        raise ValueError(f"Unknown profiles: {missing}")

    queries = _load_queries(Path(args.queries))
    if not queries:
        raise ValueError("No queries loaded.")

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.outdir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_results: List[Dict[str, Any]] = []
    run_meta = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "query_file": str(Path(args.queries)),
        "query_count": len(queries),
        "profile_config": str(profile_file),
        "profiles": selected_profiles,  # backward compatibility
        "profiles_planned": selected_profiles,
        "profile_failure_policy": "timeout_then_continue",
        "profiles_results": profile_results,
    }
    run_meta_path = out_dir / "run_meta.json"
    _write_run_meta(run_meta_path, run_meta)

    profile_summaries = []
    completed = 0
    failed = 0
    timed_out = 0
    for profile_name in selected_profiles:
        profile_overrides = deepcopy(profiles_map[profile_name] or {})
        profile_description = profile_overrides.pop("description", "")
        profile_cfg = _resolve_profile_config(
            base_config=base_config,
            defaults=defaults,
            profile_overrides=profile_overrides,
            fixed_k=args.fixed_k,
            fixed_top_n=args.fixed_top_n,
        )
        profile_timeout_sec = _safe_int(
            (profile_cfg.get("ablation", {}) or {}).get("profile_timeout_sec", 0),
            default=0,
        )
        if args.profile_timeout_sec is not None:
            profile_timeout_sec = max(0, int(args.profile_timeout_sec))

        status_row = {
            "profile": profile_name,
            "description": profile_description,
            "status": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "timeout_sec": profile_timeout_sec,
            "records": 0,
            "error": None,
        }
        profile_results.append(status_row)
        _write_run_meta(run_meta_path, run_meta)

        print(
            f"[Ablation] Running profile={profile_name} "
            f"queries={len(queries)} timeout_sec={profile_timeout_sec}"
        )

        result = _run_profile_with_timeout(
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            queries=queries,
            out_dir=out_dir,
            timeout_sec=profile_timeout_sec,
        )
        status_row["completed_at"] = _utc_now()

        result_status = result.get("status", "failed")
        if result_status == "completed":
            summary = dict(result.get("summary", {}))
            if profile_description:
                summary["description"] = profile_description
            profile_summaries.append(summary)
            status_row["status"] = "completed"
            status_row["records"] = int(summary.get("records", 0))
            status_row["backend_mode_effective"] = summary.get("backend_mode_effective", "unknown")
            status_row["backend_mode_required"] = summary.get("backend_mode_required", "unknown")
            status_row["backend_mode_config"] = summary.get("backend_mode_config", "unknown")
            status_row["backend_enforcement_passed"] = bool(
                summary.get("backend_enforcement_passed", False)
            )
            status_row["config_hash"] = summary.get("config_hash", "")
            completed += 1
        elif result_status == "timed_out":
            status_row["status"] = "timed_out"
            status_row["error"] = result.get("error")
            timed_out += 1
        else:
            status_row["status"] = "failed"
            status_row["error"] = result.get("error")
            tb = result.get("traceback")
            if tb:
                status_row["traceback"] = tb
            failed += 1

        _write_run_meta(run_meta_path, run_meta)

    run_meta["completed_at"] = _utc_now()
    run_meta["profile_summaries"] = profile_summaries
    run_meta["counts"] = {
        "planned": len(selected_profiles),
        "completed": completed,
        "failed": failed,
        "timed_out": timed_out,
    }
    run_meta["status"] = (
        "completed"
        if (failed == 0 and timed_out == 0)
        else "completed_with_issues"
    )
    _write_run_meta(run_meta_path, run_meta)

    print(f"[Ablation] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
