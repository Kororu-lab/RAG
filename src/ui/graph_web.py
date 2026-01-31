
import streamlit as st
import sys
import os
import contextlib
import io
import time

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the graph app
from src.agent.graph import app

st.set_page_config(
    page_title="RAG Agent Web Interface",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced RAG Agent")
st.caption("Interactive Chat with Real-time Terminal Output")

# session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls/info
with st.sidebar:
    st.header("Status")
    st.success("System Ready")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Custom Logger to redirect stdout to Streamlit
class StreamlitLogger:
    def __init__(self, original_stdout, placeholder):
        self.original_stdout = original_stdout
        self.placeholder = placeholder
        self.log_buffer = io.StringIO()

    def write(self, message):
        self.original_stdout.write(message) # Write to real terminal
        self.log_buffer.write(message)      # Write to buffer
        # Update Web UI
        # Note: calling this too frequently might cause performance issues, 
        # but for terminal logs it's usually acceptable.
        self.placeholder.code(self.log_buffer.getvalue(), language="text")

    def flush(self):
        self.original_stdout.flush()

# Chat Input
if prompt := st.chat_input("Enter your query"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response Container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Area for Raw Output (Collapsed by default, or open if preferred)
        with st.expander("Terminal Output (Real-time)", expanded=True):
            log_placeholder = st.empty()
            log_text = "Initializing...\n"
            log_placeholder.code(log_text, language="text")

        # Capture output and Run Graph
        full_response = ""
        final_answer_found = False
        
        # Save original stdout
        original_stdout = sys.stdout
        logger = StreamlitLogger(original_stdout, log_placeholder)

        try:
            sys.stdout = logger
            
            # Run Graph
            inputs = {"question": prompt, "search_count": 0}
            
            # Stream events
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Log Node transition
                    # print(f"Node '{key}' executed.") # Handled by graph.py prints usually
                    
                    if key == "generate" and "generation" in value:
                        final_generation = value["generation"]
                        # Just overwrite the placeholder with the final answer
                        message_placeholder.markdown(final_generation)
                        full_response = final_generation
                        final_answer_found = True
                        
                        # Add References
                        if "documents" in value:
                            refs = "\n\n**Reference Sources:**\n"
                            seen_refs = set()
                            for doc in value["documents"]:
                                ref_id = doc.metadata.get('ref_id', 'Unknown')
                                if ref_id not in seen_refs:
                                    level = doc.metadata.get('level', '0')
                                    prefix = "Summary" if str(level) == '1' else "Detail"
                                    # Create a link if possible (for local file, we can't easily, just text)
                                    refs += f"- `[{prefix}]` {ref_id}\n"
                                    seen_refs.add(ref_id)
                            
                            full_response += refs
                            message_placeholder.markdown(full_response)
                            
                    elif "question" in value and key == "rewrite":
                         pass # Already logged to stdout by graph.py

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
        # Save history
        if final_answer_found:
             st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
             # Fallback if no generation key (maybe filtered out)
             if not full_response:
                 full_response = "No response generated. Please check terminal output."
                 message_placeholder.warning(full_response)
             st.session_state.messages.append({"role": "assistant", "content": full_response})
