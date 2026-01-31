#!/bin/bash

# LTDB All-in-One Ingestion Script
# Runs Vision Capture, ColPali Indexing, and Text Indexing sequentially.

echo "=================================================="
echo "  Starting LTDB Data Ingestion"
echo "=================================================="

# 1. Vision Capture (HTML -> Images)
echo -e "\n[Step 1] Capturing Visual Elements (Async Playwright)..."
uv run vision/vision_ingest.py
if [ $? -ne 0 ]; then
    echo "Vision Ingestion Failed!"
    exit 1
fi

# 2. ColPali Indexing (Images -> Embeddings)
echo -e "\n[Step 2] Indexing Visual Content (ColPali)..."
uv run vision/colpali_ingest.py
if [ $? -ne 0 ]; then
    echo "ColPali Indexing Failed!"
    exit 1
fi

# 3. Matrix RAPTOR Pipeline (Text -> Hierarchical Summary -> Vector DB)
echo -e "\n[Step 3] Running Matrix RAPTOR Pipeline..."

# 3a. Level 0 Extraction (Text -> Jsonl)
echo -e "  [3a] Extracting Level 0 Data..."
uv run raptor/raptor_level0.py
if [ $? -ne 0 ]; then
    echo "Level 0 Extraction Failed!"
    exit 1
fi

# 3b. Level 1 Summarization (L0 -> L1 Summaries)
echo -e "  [3b] Generating Level 1 Summaries..."
uv run raptor/raptor_level1.py
if [ $? -ne 0 ]; then
    echo "Level 1 Summarization Failed!"
    exit 1
fi

# 3c. Final Ingestion (Jsonl -> Postgres)
echo -e "  [3c] Finalizing Ingestion to Postgres..."
uv run raptor/raptor_finalize.py
if [ $? -ne 0 ]; then
    echo "Final Ingestion Failed!"
    exit 1
fi

echo -e "\n=================================================="
echo "  Ingestion Complete!"
echo "=================================================="
