import os
import re
from typing import Dict, List


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw or "")
    return _WS_RE.sub(" ", text).strip()


def _read_text_snippet(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read(max_chars * 4)
    except Exception:
        return ""
    return _strip_html(raw)[:max_chars].strip()


def _pick_card_file(lang_dir: str) -> str:
    preferred = os.path.join(lang_dir, "0_general", "01_basic", "01_basic.html")
    if os.path.exists(preferred):
        return preferred

    for root, _, files in os.walk(lang_dir):
        for name in sorted(files):
            if name.lower().endswith(".html"):
                return os.path.join(root, name)
    return ""


def build_language_cards(doc_dir: str, language_ids: List[str], max_chars: int = 220) -> Dict[str, str]:
    """
    Build short per-language text cards from corpus files for LLM disambiguation.
    Cards are data-derived and contain no hard-coded alias mapping.
    """
    cards: Dict[str, str] = {}
    clip = max(80, int(max_chars))

    for lang_id in language_ids:
        lang_dir = os.path.join(doc_dir, lang_id)
        if not os.path.isdir(lang_dir):
            cards[lang_id] = ""
            continue

        card_file = _pick_card_file(lang_dir)
        if not card_file:
            cards[lang_id] = ""
            continue

        cards[lang_id] = _read_text_snippet(card_file, clip)

    return cards

