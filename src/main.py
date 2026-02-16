import os
import sys
import subprocess
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure Logging
logging.basicConfig(level=logging.WARNING)


def run_vector_pipeline(query, interactive=False):
    """Executes the standard Vector RAG pipeline (Process Isolation)."""
    print("\n[Pipeline] Running VECTOR RAG pipeline...")
    
    # Stage 1: Retrieval
    print("[Stage 1] Retrieving Context...")
    ret_code = subprocess.call(["uv", "run", "src/retrieve/rag_retrieve.py", query])
    
    if ret_code != 0:
        print("Retrieval failed.")
        return

    # Stage 2: Generation
    print("[Stage 2] Generating Answer...")
    subprocess.call(["uv", "run", "src/llm/rag_generate.py"])

def main():
    print("==================================================")
    print("  LTDB v1.3 Vector RAG Orchestrator")
    print("==================================================")

    while True:
        try:
            query = input("\nEnter query (or 'q' to quit): ").strip()
        except EOFError:
            break
            
        if query.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
        
        if not query:
            continue

        run_vector_pipeline(query)

if __name__ == "__main__":
    main()
