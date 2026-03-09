from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


GROUP_LABELS = {
    "single_lang_single_topic": "T1",
    "multi_lang_single_topic": "T2",
    "single_lang_multi_topic": "T3",
}
METRIC_KEYS = (
    ("doc@10", "doc_recall_at_10"),
    ("chunk@10", "chunk_recall_at_10"),
    ("mrr@10", "mrr_at_10"),
    ("doc@20", "doc_recall_at_20"),
    ("chunk@20", "chunk_recall_at_20"),
    ("mrr@20", "mrr_at_20"),
)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_macro_metrics(path: Path) -> Dict[str, float]:
    rows = _read_csv_rows(path)
    metrics: Dict[str, float] = {}
    for row in rows:
        if row.get("aggregation") != "macro":
            continue
        key = f"{row.get('metric')}@{row.get('k')}"
        try:
            metrics[key] = float(row.get("value", "0") or 0.0)
        except ValueError:
            metrics[key] = 0.0
    return metrics


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _iter_group_rows(rows: Iterable[Dict[str, str]], query_type: str) -> List[Dict[str, str]]:
    return [row for row in rows if row.get("query_type") == query_type]


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(*, run_dir: Path, query_file: str, out_dir: Path, repo_root: Path, metadata_path: str) -> None:
    run_meta_path = run_dir / "run_meta.json"
    with run_meta_path.open("r", encoding="utf-8") as f:
        run_meta = json.load(f)

    profile_dirs = [path for path in sorted(run_dir.iterdir()) if path.is_dir() and (path / "macro_micro.csv").exists()]

    main_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    profiles: List[str] = []

    for profile_dir in profile_dirs:
        profile = profile_dir.name
        profiles.append(profile)
        metrics = _read_macro_metrics(profile_dir / "macro_micro.csv")
        main_rows.append(
            {
                "profile": profile,
                "doc@10": metrics.get("doc_recall@10", 0.0),
                "chunk@10": metrics.get("chunk_recall@10", 0.0),
                "mrr@10": metrics.get("mrr@10", 0.0),
                "doc@20": metrics.get("doc_recall@20", 0.0),
                "chunk@20": metrics.get("chunk_recall@20", 0.0),
                "mrr@20": metrics.get("mrr@20", 0.0),
            }
        )

        summary_rows = _read_csv_rows(profile_dir / "summary.csv")
        for query_type, group_label in GROUP_LABELS.items():
            group_slice = _iter_group_rows(summary_rows, query_type)
            if not group_slice:
                continue
            row: Dict[str, object] = {
                "profile": profile,
                "group_label": group_label,
                "query_type": query_type,
                "query_count": len(group_slice),
            }
            for output_key, source_key in METRIC_KEYS:
                row[output_key] = mean(_safe_float(item.get(source_key, "0")) for item in group_slice)
            group_rows.append(row)

    _write_csv(
        out_dir / "main_table_metrics.csv",
        ["profile", "doc@10", "chunk@10", "mrr@10", "doc@20", "chunk@20", "mrr@20"],
        main_rows,
    )
    _write_csv(
        out_dir / "groupwise_metrics.csv",
        ["profile", "group_label", "query_type", "query_count", "doc@10", "chunk@10", "mrr@10", "doc@20", "chunk@20", "mrr@20"],
        group_rows,
    )

    profile_name_map = {
        "B0": "raw_dense_baseline",
        "B1": "summary_dense_baseline",
        "B2": "summary_dense_plus_bm25_rrf",
        "B3": "summary_dense_plus_recursive_retrieval",
        "B4": "query_only_metadata_filtering",
        "B6": "b4_plus_reranker",
        "B7": "finalized_materialized_language_split",
    }
    with (out_dir / "profile_name_map.json").open("w", encoding="utf-8") as f:
        json.dump(profile_name_map, f, ensure_ascii=False, indent=2)

    manifest = {
        "run_id": run_dir.name,
        "query_file": query_file,
        "profiles": profiles,
        "timestamp": run_meta.get("started_at"),
        "repo_path": str(repo_root.resolve()),
        "metadata_file_path": metadata_path,
        "run_meta_path": str(run_meta_path),
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--metadata-path", required=True)
    args = parser.parse_args()

    build_outputs(
        run_dir=Path(args.run_dir),
        query_file=args.query_file,
        out_dir=Path(args.out_dir),
        repo_root=Path(args.repo_root),
        metadata_path=args.metadata_path,
    )


if __name__ == "__main__":
    main()
