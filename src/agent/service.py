from typing import Any, Dict, Iterator

from src.agent.graph import app


def run_query_stream(question: str, search_count: int = 0) -> Iterator[Dict[str, Dict[str, Any]]]:
    """Stream graph events for a single query."""
    inputs = {"question": question, "search_count": search_count}
    for event in app.stream(inputs):
        yield event


def run_query(question: str, search_count: int = 0) -> Dict[str, Dict[str, Any]]:
    """Run graph to completion and return the latest payload per node key."""
    last_payloads: Dict[str, Dict[str, Any]] = {}
    for event in run_query_stream(question, search_count=search_count):
        for node_key, payload in event.items():
            last_payloads[node_key] = payload
    return last_payloads
