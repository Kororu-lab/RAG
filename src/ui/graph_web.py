"""RAG Agent Web Interface - Streamlit UI for testing RAG system."""

import streamlit as st
import sys
import os
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent.graph import app

st.set_page_config(
    page_title="RAG Agent Web Interface",
    page_icon="🤖",
    layout="wide"
)

st.title("테스트용 RAG(Agent)")
st.caption("src/agent/graph.py 기반")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model Selection
    model_options = ["gpt-oss:20b", "gpt-oss:120b"]
    selected_model = st.selectbox("Select LLM Model", model_options, index=0)
    st.info(f"Current Model: **{selected_model}**")
    
    st.divider()
    
    # Retrieval Settings
    st.header("⚙️ Retrieval Settings")
    bm25_enabled = st.checkbox("BM25 Hybrid Search", value=True)
    reranker_enabled = st.checkbox("Reranker", value=True)
    recursive_enabled = st.checkbox("Recursive Retrieval", value=True)
    vision_enabled = st.checkbox("Vision Search", value=False)
    
    st.divider()
    
    # Config Patching
    import src.utils
    
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
