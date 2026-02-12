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

from src.agent.graph import app
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
    
    st.divider()
    
    # Config Patching
    
    if not hasattr(src.utils, "_original_load_config"):
        src.utils._original_load_config = src.utils.load_config

    def patched_load_config():
        config = src.utils._original_load_config()
        
        # Model override
        config.setdefault("llm_retrieval", {})["model_name"] = selected_model
        if "llm" in config:
            config["llm"]["model_name"] = selected_model
        
        # Retrieval settings override
        retrieval = config.setdefault("retrieval", {})
        retrieval.setdefault("hybrid_search", {})["enabled"] = bm25_enabled
        retrieval["recursive_retrieval"] = recursive_enabled
        retrieval["vision_search"] = vision_enabled
        
        # Reranker override
        config.setdefault("reranker", {})["enabled"] = reranker_enabled
        config.setdefault("rag", {})["generation_timeout_sec"] = int(generation_timeout_sec)
        
        return config

    src.utils.load_config = patched_load_config
    
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
        
        with st.expander("Terminal Output (Real-time)", expanded=True):
            log_placeholder = st.empty()
            log_placeholder.code("Initializing...\n", language="text")

        full_response = ""
        final_answer_found = False
        original_stdout = sys.stdout
        logger = StreamlitLogger(original_stdout, log_placeholder)

        try:
            sys.stdout = logger
            inputs = {"question": prompt, "search_count": 0}
            
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "generate" and "generation" in value:
                        final_generation = value["generation"]
                        message_placeholder.markdown(final_generation)
                        full_response = final_generation
                        final_answer_found = True
                        
                        if "documents" in value:
                            refs = "\n\n**Reference Sources:**\n"
                            seen_refs = set()
                            for doc in value["documents"]:
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
                                for doc in value["documents"]:
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

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            
        if final_answer_found:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            if not full_response:
                full_response = "No response generated. Please check terminal output."
                message_placeholder.warning(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
