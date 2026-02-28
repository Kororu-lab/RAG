from __future__ import annotations

import json
import time
from typing import Callable, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def parse_json_strict(raw_text: str):
    """Parse model output as strict JSON only."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("json_empty_response")
    return json.loads(text)


def repair_json_text(
    *,
    llm,
    raw_text: str,
    schema_hint: str,
    retries: int = 1,
    backoff_sec: float = 0.25,
    timeout_checker: Callable[[Exception], bool] | None = None,
    on_timeout: Callable[[str], None] | None = None,
    timeout_location: str = "json_repair",
) -> Tuple[str, int, bool]:
    """
    Ask the LLM to rewrite prior output into strict JSON.
    Returns: (last_text, retries_used, is_valid_json)
    """
    attempts = max(0, int(retries))
    if attempts == 0:
        return (raw_text or "").strip(), 0, False

    prompt = ChatPromptTemplate.from_template(
        """Convert the following model output into valid JSON.
Return JSON only. Do not include markdown fences. Do not include prose.

Schema hint:
{schema_hint}

Original output:
{raw_output}

JSON:"""
    )
    chain = prompt | llm | StrOutputParser()
    last_text = (raw_text or "").strip()

    for attempt in range(1, attempts + 1):
        try:
            candidate = (chain.invoke(
                {
                    "schema_hint": schema_hint,
                    "raw_output": last_text,
                }
            ) or "").strip()
            last_text = candidate
        except Exception as exc:
            if timeout_checker and timeout_checker(exc) and on_timeout:
                on_timeout(timeout_location)
            if attempt < attempts:
                sleep_sec = max(0.0, float(backoff_sec)) * attempt
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            continue

        try:
            json.loads(last_text)
            return last_text, attempt, True
        except Exception:
            if attempt < attempts:
                sleep_sec = max(0.0, float(backoff_sec)) * attempt
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

    return last_text, attempts, False
