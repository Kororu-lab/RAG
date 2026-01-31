#!/bin/bash

# LTDB All-in-One Ingestion Script
# Runs Vision Capture, ColPali Indexing, and Text Indexing sequentially.
# Assumes running from repository root (RAG/)

echo "=================================================="
echo "  Starting LTDB Data Ingestion"
echo "=================================================="

# Ensure we are in the project root by checking for config.yaml
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found. Please run this script from the repository root (e.g. ./src/ingest/run_ingest.sh)"
    exit 1
fi

# 1. Vision Capture (HTML -> Images)
echo -e "\n[Step 1] Capturing Visual Elements (Async Playwright)..."
uv run src/ingest/vision/vision_ingest.py
if [ $? -ne 0 ]; then
    echo "Vision Ingestion Failed!"
    exit 1
fi

# 2. ColPali Indexing (Images -> Embeddings)
# Note: Ensure valid languages logic in graph_ingest doesn't conflict or if it's needed here.
echo -e "\n[Step 2] Indexing Visual Content (ColPali)..."
uv run src/ingest/vision/colpali_ingest.py
if [ $? -ne 0 ]; then
    echo "ColPali Indexing Failed!"
    exit 1
fi

# 3. Graph Ingestion (Optional but good to have in sequence if graph is enabled)
# Since the user specifically asked for graph/ folder, we include it.
echo -e "\n[Step 3] Building Knowledge Graph..."
uv run src/ingest/graph/graph_ingest.py
if [ $? -ne 0 ]; then
    echo "Graph Ingestion Failed! (Continuing pipeline as it might be optional)"
fi

# 4. Matrix RAPTOR Pipeline (Text -> Hierarchical Summary -> Vector DB)
echo -e "\n[Step 4] Running Matrix RAPTOR Pipeline..."

# 4a. Level 0 Extraction (Text -> Jsonl)
echo -e "  [4a] Extracting Level 0 Data..."
uv run src/ingest/raptor/raptor_level0.py
if [ $? -ne 0 ]; then
    echo "Level 0 Extraction Failed!"
    exit 1
fi

# 4b. Level 1 Summarization (L0 -> L1 Summaries)
echo -e "  [4b] Generating Level 1 Summaries..."
uv run src/ingest/raptor/raptor_level1.py
if [ $? -ne 0 ]; then
    echo "Level 1 Summarization Failed!"
    exit 1
fi

# 4c. Final Ingestion (Jsonl -> Postgres)
echo -e "  [4c] Finalizing Ingestion to Postgres..."
uv run src/ingest/raptor/raptor_finalize.py
if [ $? -ne 0 ]; then
    echo "Final Ingestion Failed!"
    exit 1
fi

echo -e "\n=================================================="
echo "  Ingestion Complete!"
echo "=================================================="
