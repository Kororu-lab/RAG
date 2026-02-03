# Data Directory

This directory stores the linguistic database and generated ingestion artifacts.
**Note**: The actual data files are not included in the repository (git-ignored) due to size constraints.

## Directory Structure
- **`ltdb/`**: The raw source data (HTML files).
  - Structure: `doc/{Language}/{Topic}/{File}.html`
  - Input for the ingestion pipeline.
- **`raptor/`**: Generated text summaries from the RAPTOR pipeline.
  - `matrix_raptor_L0_data.jsonl`: Cleaned chunk-level summaries.
  - `matrix_raptor_L1_data.jsonl`: Aggregated higher-level summaries (Language/Topic overviews).
- **`ltdb_vision/`**: Extracted visual element metadata (images, tables) from the source documents.
- **`ltdb_colpali/`**: Generated ColPali embeddings and indices for visual retrieval.

## Setup
To run the system, place your `ltdb` (or `doc`) folder inside this directory or configure the path in `config.yaml`.
Then run the ingestion script:
```bash
../src/ingest/run_ingest.sh
```
