from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any
import os
import sys

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

# Ensure repository root is importable when executed as a script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils import load_config

LANG_PATCH_FIELDS = ("glottocode", "iso639_3", "lang_family", "region", "region_detail")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _parse_doc_path(path: str) -> tuple[str, str, str] | None:
    raw = _norm(path)
    if not raw:
        return None
    parts = raw.split("/")
    if len(parts) < 4:
        return None
    if parts[0] != "doc":
        return None
    return parts[1], parts[2], parts[3]


def _derive_single_lang_from_child_files(child_files: list[str] | None) -> str:
    langs = {
        parsed[0]
        for path in (child_files or [])
        if (parsed := _parse_doc_path(path)) is not None and parsed[0]
    }
    if len(langs) == 1:
        return next(iter(langs))
    return ""


def _derive_single_topic_pair_from_child_files(child_files: list[str] | None) -> tuple[str, str] | None:
    pairs = {
        (parsed[1], parsed[2])
        for path in (child_files or [])
        if (parsed := _parse_doc_path(path)) is not None and parsed[1] and parsed[2]
    }
    if len(pairs) == 1:
        return next(iter(pairs))
    return None


def _set_if_missing(
    updates: dict[str, Any],
    metadata: dict[str, Any],
    key: str,
    value: Any,
    *,
    overwrite: bool,
) -> bool:
    val = _norm(value) if isinstance(value, str) or value is not None else ""
    if val == "":
        return False

    current = metadata.get(key)
    current_norm = _norm(current)
    if current_norm and not overwrite:
        return False
    if current_norm == val:
        return False

    updates[key] = val
    return True


def _derive_l1_fields_conservative(metadata: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    row_type = _norm(metadata.get("type"))
    group_key = _norm(metadata.get("group_key"))
    child_files = metadata.get("child_files") or []

    # lang_id: existing wins; infer only when a single value is provable.
    if not _norm(metadata.get("lang_id")):
        if row_type == "L1_lang" and group_key and "/" not in group_key:
            updates["lang_id"] = group_key
        else:
            lang_id = _derive_single_lang_from_child_files(child_files)
            if lang_id:
                updates["lang_id"] = lang_id

    # parent/child topic ids: infer only from a single consistent topic pair.
    missing_parent = not _norm(metadata.get("parent_topic_id"))
    missing_child = not _norm(metadata.get("child_topic_id"))
    if missing_parent or missing_child:
        pair = _derive_single_topic_pair_from_child_files(child_files)
        if pair is not None:
            parent_topic_id, child_topic_id = pair
            if missing_parent:
                updates["parent_topic_id"] = parent_topic_id
            if missing_child:
                updates["child_topic_id"] = child_topic_id

    return updates


def build_metadata_patch(
    metadata: dict[str, Any],
    *,
    lang_index: dict[str, dict[str, str]],
    patch_version: str,
    patch_source: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    level = _norm(metadata.get("level"))
    row_type = _norm(metadata.get("type"))

    # Conservative L1 inference only.
    if level == "1":
        derived_l1 = _derive_l1_fields_conservative(metadata)
        for key, value in derived_l1.items():
            _set_if_missing(updates, metadata, key, value, overwrite=overwrite)

    # L0 topic derivation from source_file when unambiguous.
    if level == "0":
        parsed = _parse_doc_path(_norm(metadata.get("source_file")))
        if parsed is not None:
            lang_id, parent_topic_id, child_topic_id = parsed
            _set_if_missing(updates, metadata, "lang_id", lang_id, overwrite=overwrite)
            _set_if_missing(
                updates,
                metadata,
                "parent_topic_id",
                parent_topic_id,
                overwrite=overwrite,
            )
            _set_if_missing(
                updates,
                metadata,
                "child_topic_id",
                child_topic_id,
                overwrite=overwrite,
            )

    effective_lang_id = _norm(updates.get("lang_id") or metadata.get("lang_id"))
    if effective_lang_id:
        _set_if_missing(updates, metadata, "lang", effective_lang_id, overwrite=overwrite)

        lang_meta = lang_index.get(effective_lang_id, {})
        for field in LANG_PATCH_FIELDS:
            _set_if_missing(
                updates,
                metadata,
                field,
                _norm(lang_meta.get(field)),
                overwrite=overwrite,
            )
        _set_if_missing(
            updates,
            metadata,
            "family_id",
            _norm(lang_meta.get("lang_family")),
            overwrite=overwrite,
        )

    parent_topic_id = _norm(updates.get("parent_topic_id") or metadata.get("parent_topic_id"))
    child_topic_id = _norm(updates.get("child_topic_id") or metadata.get("child_topic_id"))
    if parent_topic_id and child_topic_id:
        pair = f"{parent_topic_id}/{child_topic_id}"
        _set_if_missing(updates, metadata, "topic_pair_id", pair, overwrite=overwrite)

    if updates:
        updates["metadata_patch_version"] = patch_version
        updates["metadata_patch_source"] = patch_source

    return updates


def load_metadata_index(metadata_csv: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lang_id = _norm(row.get("lang_id"))
            if not lang_id:
                continue
            index[lang_id] = {
                "glottocode": _norm(row.get("glottocode")),
                "iso639_3": _norm(row.get("iso639_3")),
                "lang_family": _norm(row.get("lang_family")),
                "region": _norm(row.get("region")),
                "region_detail": _norm(row.get("region_detail")),
            }
    return index


def _db_connect() -> tuple[Any, str]:
    config = load_config()
    db_cfg = config["database"]
    table_name = db_cfg["table_name"]

    conn = psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        dbname=db_cfg["dbname"],
    )
    return conn, table_name


def run_patch(
    *,
    metadata_csv: Path,
    table_name: str | None,
    patch_version: str,
    patch_source: str,
    overwrite: bool,
    execute: bool,
) -> None:
    lang_index = load_metadata_index(metadata_csv)

    conn, cfg_table_name = _db_connect()
    target_table = table_name or cfg_table_name

    changed_rows = 0
    changed_keys: Counter = Counter()
    updates_to_apply: list[tuple[int, dict[str, Any]]] = []
    rows_scanned = 0

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id, metadata FROM {} ORDER BY id").format(
                    sql.Identifier(target_table)
                )
            )
            for row_id, metadata in cur.fetchall():
                rows_scanned += 1
                if not isinstance(metadata, dict):
                    continue

                patch = build_metadata_patch(
                    metadata,
                    lang_index=lang_index,
                    patch_version=patch_version,
                    patch_source=patch_source,
                    overwrite=overwrite,
                )
                if not patch:
                    continue

                changed_rows += 1
                for key in patch:
                    changed_keys[key] += 1
                updates_to_apply.append((row_id, patch))

        print(f"[metadata-patch] table={target_table}")
        print(f"[metadata-patch] rows_scanned={rows_scanned} patches={changed_rows}")
        print(f"[metadata-patch] changed_keys={dict(changed_keys)}")
        print(f"[metadata-patch] execute={execute}")

        if not execute:
            conn.rollback()
            return

        with conn.cursor() as cur:
            stmt = sql.SQL(
                "UPDATE {} SET metadata = coalesce(metadata, '{{}}'::jsonb) || %s::jsonb WHERE id = %s"
            ).format(sql.Identifier(target_table))
            for row_id, patch in updates_to_apply:
                cur.execute(stmt, (Json(patch), row_id))
        conn.commit()
        print("[metadata-patch] commit=ok")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch DB metadata JSONB from metadata CSV.")
    parser.add_argument("--metadata-csv", default="data/metadata.enriched.csv")
    parser.add_argument(
        "--patch-version",
        default="2026-03-04.enrich_v1",
        help="Value for metadata_patch_version.",
    )
    parser.add_argument(
        "--patch-source",
        default="metadata.enriched.csv",
        help="Value for metadata_patch_source.",
    )
    parser.add_argument("--table-name", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply updates. Without this flag, runs as dry-run.",
    )
    args = parser.parse_args()

    run_patch(
        metadata_csv=Path(args.metadata_csv),
        table_name=_norm(args.table_name) or None,
        patch_version=args.patch_version,
        patch_source=args.patch_source,
        overwrite=args.overwrite,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
