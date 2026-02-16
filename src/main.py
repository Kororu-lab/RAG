import os
import sys
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.service import run_query

# Configure Logging
logging.basicConfig(level=logging.WARNING)


def run_vector_pipeline(query, interactive=False):
    """Executes the unified graph service pipeline."""
    print("\n[Pipeline] Running GRAPH RAG pipeline...")
    results = run_query(query, search_count=0)

    final_payload = results.get("check_hallucination") or results.get("generate") or {}
    generation = final_payload.get("generation")

    if generation:
        print("\n[Final Answer]")
        print(generation)
    else:
        print("No answer generated.")

    documents = final_payload.get("documents")
    if not documents:
        documents = (results.get("generate") or {}).get("documents", [])

    if documents:
        print("\n[References]")
        unique_refs = []
        seen_refs = set()
        for doc in documents:
            ref_id = doc.metadata.get("ref_id", "Unknown")
            if ref_id in seen_refs:
                continue
            level = doc.metadata.get("level", "0")
            prefix = "Summary" if str(level) == "1" else "Detail"
            seen_refs.add(ref_id)
            unique_refs.append((prefix, ref_id))

        for prefix, ref_id in unique_refs[:10]:
            print(f"- [{prefix}] {ref_id}")
        if len(unique_refs) > 10:
            print(f"- ... and {len(unique_refs) - 10} more")

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
