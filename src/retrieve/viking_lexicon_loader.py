"""Load Viking lexicon from external YAML file with caching."""

import os
import yaml
from typing import Dict, List, Optional, Tuple

_cached_lexicon: Optional[Tuple[Dict[str, List[str]], Dict[str, str]]] = None
_cached_path: Optional[str] = None


def load_viking_lexicon(
    lexicon_path: str = "config/viking_lexicon.yaml",
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Load phenomenon and category lexicons from YAML.

    Returns (phenomenon_lexicon, category_lexicon).
    Caches after first load; returns cached copy on subsequent calls
    for the same path.
    """
    global _cached_lexicon, _cached_path
    if _cached_lexicon is not None and _cached_path == lexicon_path:
        return _cached_lexicon

    resolved = os.path.abspath(lexicon_path)
    if not os.path.isfile(resolved):
        print(f"[VikingLexicon] File not found: {resolved} — using empty lexicon")
        _cached_lexicon = ({}, {})
        _cached_path = lexicon_path
        return _cached_lexicon

    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    phenomenon: Dict[str, List[str]] = {}
    for term, codes in (data.get("phenomenon") or {}).items():
        if isinstance(codes, list):
            phenomenon[str(term)] = [str(c) for c in codes]
        else:
            phenomenon[str(term)] = [str(codes)]

    category: Dict[str, str] = {}
    for term, code in (data.get("category") or {}).items():
        category[str(term)] = str(code)

    _cached_lexicon = (phenomenon, category)
    _cached_path = lexicon_path
    print(f"[VikingLexicon] Loaded: {len(phenomenon)} phenomenon terms, "
          f"{len(category)} category terms from {resolved}")
    return _cached_lexicon


def invalidate_cache() -> None:
    """Force reload on next call (useful for hot-reload in tests)."""
    global _cached_lexicon, _cached_path
    _cached_lexicon = None
    _cached_path = None
