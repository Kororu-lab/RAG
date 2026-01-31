import sys
import os
import logging

# Suppress logs for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.agent.graph import app

def visualize():
    try:
        print("Generating Graph PNG...")
        # get_graph(xray=True) helps to see subgraphs if any, though here it is flat.
        png_bytes = app.get_graph().draw_mermaid_png()
        
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_structure.png"))
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph saved to {output_path}")
    except Exception as e:
        print(f"Error drawing graph: {e}")
        # print("Ensure 'grandalf' or 'pygraphviz' is installed if draw_mermaid_png fails.")

if __name__ == "__main__":
    visualize()
