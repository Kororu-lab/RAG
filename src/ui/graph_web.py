"""RAG Agent Web Interface - Streamlit UI for testing RAG system."""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import io
from typing import Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent.service import run_query_stream
from src.ui.model_options import DEFAULT_MODEL, MODEL_OPTIONS

LTDB_ROOT = os.path.abspath(os.path.join(project_root, "data", "ltdb"))
# Kept as a dedicated variable for easy future extension (e.g., graded docs mode).
HTML_SOURCE_SCOPE = "retrieved_refs_only"


def _resolve_ltdb_source_path(source_file: str) -> Tuple[Optional[str], Optional[str]]:
    if not source_file:
        return None, "Missing source_file metadata."

    normalized = str(source_file).replace("\\", "/").lstrip("/")
    if normalized.startswith("data/ltdb/"):
        normalized = normalized[len("data/ltdb/") :]

    resolved = os.path.abspath(os.path.join(LTDB_ROOT, normalized))
    base = os.path.abspath(LTDB_ROOT)
    if not (resolved == base or resolved.startswith(base + os.sep)):
        return None, "Blocked unsafe source path."

    if not os.path.isfile(resolved):
        return None, f"Original HTML not found: {resolved}"

    return resolved, None


def _load_original_html(source_file: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    resolved, err = _resolve_ltdb_source_path(source_file)
    if err:
        return None, err, resolved
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), None, resolved
    except Exception as e:
        return None, f"Failed to read HTML: {e}", resolved


st.set_page_config(
    page_title="RAG Agent Web Interface",
    page_icon="🤖",
    layout="wide"
)

st.title("테스트용 RAG(Agent)")
st.caption("src/agent/graph.py 기반")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extra_model_options" not in st.session_state:
    st.session_state.extra_model_options = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model Selection
    import src.utils
    model_options = list(dict.fromkeys(MODEL_OPTIONS + st.session_state.extra_model_options))
    default_index = model_options.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_options else 0

    selected_model = st.selectbox("Select LLM Model", model_options, index=default_index)
    custom_model = st.text_input("Add custom model option", placeholder="e.g., gemma3:27b")
    if st.button("Add Model") and custom_model.strip():
        new_model = custom_model.strip()
        if new_model not in st.session_state.extra_model_options and new_model not in model_options:
            st.session_state.extra_model_options.append(new_model)
            st.rerun()
    
    st.info(f"Current Model: **{selected_model}**")
    
    st.divider()
    
    # Retrieval Settings
    st.header("⚙️ Retrieval Settings")
    bm25_enabled = st.checkbox("BM25 Hybrid Search", value=True)
    reranker_enabled = st.checkbox("Reranker", value=True)
    recursive_enabled = st.checkbox("Recursive Retrieval", value=True)
    vision_enabled = st.checkbox("Vision Search", value=False)
    generation_timeout_sec = st.number_input("Generation Timeout (sec)", min_value=10, max_value=600, value=60, step=5)
    show_original_html = st.checkbox("Show Original HTML for Retrieved Refs", value=True)

    st.subheader("Viking Routing")
    viking_enabled = st.checkbox("Viking Routing", value=False)
    viking_soft = st.checkbox("Viking Soft Fallback", value=True)
    viking_max_exp = st.number_input("Viking Max Expansions", min_value=0, max_value=5, value=2, step=1)

    st.subheader("Pipeline Controls")
    grading_enabled = st.checkbox("Document Grading", value=True)
    hallucination_check_enabled = st.checkbox("Hallucination Check", value=True)

    st.divider()
    
    def build_runtime_config():
        config = src.utils.load_config()

        # Model override
        config.setdefault("llm_retrieval", {})["model_name"] = selected_model
        if "llm" in config:
            config["llm"]["model_name"] = selected_model

        # Retrieval settings override
        retrieval = config.setdefault("retrieval", {})
        retrieval.setdefault("hybrid_search", {})["enabled"] = bm25_enabled
        retrieval["recursive_retrieval"] = recursive_enabled
        retrieval["vision_search"] = vision_enabled

        # Viking routing override
        viking = retrieval.setdefault("viking", {})
        viking["enabled"] = viking_enabled
        viking["mode"] = "soft" if viking_soft else "strict"
        viking["max_expansions"] = int(viking_max_exp)

        # Reranker override
        config.setdefault("reranker", {})["enabled"] = reranker_enabled
        rag = config.setdefault("rag", {})
        rag["generation_timeout_sec"] = int(generation_timeout_sec)
        rag["skip_grading"] = not grading_enabled
        rag["skip_hallucination"] = not hallucination_check_enabled

        return config
    
    st.header("Status")
    st.success("System Ready")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


class StreamlitLogger:
    """Redirects stdout to Streamlit code block."""
    
    def __init__(self, original_stdout, placeholder):
        self.original_stdout = original_stdout
        self.placeholder = placeholder
        self.log_buffer = io.StringIO()

    def write(self, message):
        self.original_stdout.write(message)
        self.log_buffer.write(message)
        self.placeholder.code(self.log_buffer.getvalue(), language="text")

    def flush(self):
        self.original_stdout.flush()


# Chat Input
if prompt := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        full_response = ""
        final_answer_found = False
        generate_output = None
        original_stdout = sys.stdout

        with st.status("Running pipeline...", expanded=True) as pipeline_status:
            step_placeholder = st.empty()
            step_log = []
            pipeline_errored = False
            strict_zero_doc_warned = False

            with st.expander("Terminal Output", expanded=False):
                log_placeholder = st.empty()
                log_placeholder.code("Initializing...\n", language="text")

            logger = StreamlitLogger(original_stdout, log_placeholder)

            try:
                sys.stdout = logger
                runtime_cfg = build_runtime_config()
                with src.utils.use_config_override(runtime_cfg):
                    for output in run_query_stream(prompt, search_count=0):
                        for key, value in output.items():
                            if key == "retrieve":
                                n = len(value.get("documents", []))
                                viking_tag = ""
                                if viking_enabled:
                                    viking_tag = " (Viking)"
                                    # Extract scope details from first doc metadata
                                    docs = value.get("documents", [])
                                    scope_info = None
                                    for d in docs:
                                        scope_info = d.metadata.get("_viking_scope")
                                        if scope_info:
                                            break
                                    if scope_info:
                                        parts = []
                                        if scope_info.get("languages"):
                                            parts.append(f"lang={scope_info['languages']}")
                                        if scope_info.get("phenomena"):
                                            parts.append(f"phenomena={scope_info['phenomena']}")
                                        elif scope_info.get("categories"):
                                            parts.append(f"categories={scope_info['categories']}")
                                        parts.append(f"conf={scope_info.get('confidence', '?')}")
                                        parts.append(f"patterns={scope_info.get('patterns', 0)}")
                                        viking_tag += f" `[{', '.join(parts)}]`"
                                        trace = scope_info.get("trace", [])
                                        if any("widened" in t for t in trace):
                                            viking_tag += " *(fallback)*"
                                step_log.append(f"**Retrieve**{viking_tag} — {n} documents")
                                if viking_enabled and (not viking_soft) and n == 0:
                                    step_log.append(
                                        "**Viking Strict Warning** — 0 docs in current strict scope. "
                                        "Try soft mode, increase max expansions, or tune min_hits."
                                    )
                                    if not strict_zero_doc_warned:
                                        st.warning(
                                            "Viking strict scope returned 0 documents. "
                                            "Suggested actions: switch to soft mode, "
                                            "increase Viking Max Expansions, adjust min_hits."
                                        )
                                        strict_zero_doc_warned = True
                                if grading_enabled:
                                    pipeline_status.update(label="Grading documents...")
                                else:
                                    pipeline_status.update(label="Generating answer...")

                            elif key == "grade_documents":
                                n = len(value.get("documents", []))
                                if grading_enabled:
                                    step_log.append(f"**Grade Documents** — {n} relevant")
                                else:
                                    step_log.append(f"**Grade Documents** — skipped (all {n} passed)")
                                pipeline_status.update(label="Generating answer...")

                            elif key == "rewrite":
                                q = value.get("question", "")
                                count = value.get("search_count", "?")
                                step_log.append(f"**Rewrite Query** (attempt {count}) — {q[:100]}")
                                pipeline_status.update(label="Re-retrieving...")

                            elif key == "generate" and "generation" in value:
                                step_log.append("**Generate** — Answer produced")
                                if hallucination_check_enabled:
                                    pipeline_status.update(label="Checking hallucination...")
                                else:
                                    pipeline_status.update(label="Finalizing...")
                                generate_output = value
                                full_response = value["generation"]
                                final_answer_found = True
                                message_placeholder.markdown(full_response)

                            elif key == "check_hallucination":
                                if hallucination_check_enabled:
                                    h = value.get("hallucination_status", "unknown")
                                    label = "GROUNDED" if h == "pass" else "FAILED"
                                    step_log.append(f"**Hallucination Check** — {label}")
                                else:
                                    step_log.append("**Hallucination Check** — skipped")
                                if "generation" in value:
                                    full_response = value["generation"]
                                    generate_output = value
                                    message_placeholder.markdown(full_response)

                            step_placeholder.markdown("\n".join(f"- {s}" for s in step_log))

            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                traceback.print_exc()
                pipeline_errored = True
                pipeline_status.update(label="Pipeline error", state="error")
            finally:
                sys.stdout = original_stdout

            if not pipeline_errored:
                if final_answer_found:
                    pipeline_status.update(label="Pipeline complete", state="complete", expanded=False)
                else:
                    pipeline_status.update(label="Pipeline finished (no answer)", state="complete")

        # Display references and original HTML after pipeline completes
        if final_answer_found and generate_output and "documents" in generate_output:
            refs = "\n\n**Reference Sources:**\n"
            seen_refs = set()
            for doc in generate_output["documents"]:
                ref_id = doc.metadata.get('ref_id', 'Unknown')
                if ref_id not in seen_refs:
                    level = doc.metadata.get('level', '0')
                    prefix = "Summary" if str(level) == '1' else "Detail"
                    refs += f"- `[{prefix}]` {ref_id}\n"
                    seen_refs.add(ref_id)

            full_response += refs
            message_placeholder.markdown(full_response)

            if show_original_html and HTML_SOURCE_SCOPE == "retrieved_refs_only":
                st.markdown("**Original Source HTML (Retrieved Refs):**")
                seen_sources = set()
                for doc in generate_output["documents"]:
                    source_file = str(doc.metadata.get("source_file", "")).strip()
                    ref_id = doc.metadata.get("ref_id", "Unknown")
                    if not source_file or source_file in seen_sources:
                        continue
                    seen_sources.add(source_file)

                    with st.expander(f"{ref_id} | {source_file}", expanded=False):
                        html_text, html_err, resolved_path = _load_original_html(source_file)
                        if html_err:
                            st.warning(html_err)
                            if resolved_path:
                                st.caption(f"Resolved path: `{resolved_path}`")
                        else:
                            components.html(html_text, height=500, scrolling=True)
                            st.caption(f"Source: `{source_file}`")

        if final_answer_found:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            if not full_response:
                full_response = "No response generated. Please check terminal output."
                message_placeholder.warning(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
