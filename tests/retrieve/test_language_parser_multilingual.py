import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieve import rag_retrieve


def test_parse_detected_languages_supports_multilingual_separators():
    known_languages = ["achang", "takpa", "korean"]

    assert rag_retrieve.parse_detected_languages(
        "achang and takpa", known_languages
    ) == ["achang", "takpa"]
    assert rag_retrieve.parse_detected_languages(
        "takpa/achang", known_languages
    ) == ["takpa", "achang"]


def test_parse_detected_languages_embedded_detection_with_single_candidate():
    known_languages = ["achang", "takpa", "korean"]

    parsed = rag_retrieve.parse_detected_languages("takpa achang", known_languages)

    assert parsed == ["takpa", "achang"]


def test_parse_detected_languages_preserves_none_null_and_comma_behavior():
    known_languages = ["achang", "takpa", "korean"]

    assert rag_retrieve.parse_detected_languages("", known_languages) is None
    assert rag_retrieve.parse_detected_languages("none", known_languages) is None
    assert rag_retrieve.parse_detected_languages(" null ", known_languages) is None
    assert rag_retrieve.parse_detected_languages(
        "achang, takpa, achang", known_languages
    ) == ["achang", "takpa"]
