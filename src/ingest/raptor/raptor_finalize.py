import argparse
import hashlib
import os
import sys
import json
import psycopg2
from psycopg2 import sql
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from psycopg2.extras import Json

# Ensure src is resolvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils import load_config, LLMUtility, resolve_torch_device, clear_torch_cache

def get_db_connection(config: Dict):
    """Establishes connection to PostgreSQL."""
    db_cfg = config['database']
    conn = psycopg2.connect(
        host=db_cfg['host'],
        port=db_cfg['port'],
        user=db_cfg['user'],
        password=db_cfg['password'],
        dbname=db_cfg['dbname']
    )
    return conn

def build_doc_key(item: Dict[str, Any]) -> str:
    """Builds a stable unique key for upsert safety across ingestion runs."""
    source_file = str(item.get("source_file", "")).strip()
    chunk_id = item.get("chunk_id")
    level = str(item.get("level", "")).strip()
    node_type = str(item.get("type", "")).strip()
    group_key = str(item.get("group_key", "")).strip()

    if level == "0" and source_file and chunk_id is not None:
        return f"L0::{source_file}::{chunk_id}"
    if level == "1" and node_type and group_key:
        return f"L1::{node_type}::{group_key}"

    serialized = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256::{digest}"

def ensure_table(conn, table_name: str, vector_dim: int = 1024, rebuild: bool = False):
    """Ensures pgvector table/indexes exist; optional rebuild for first deployment."""
    vector_dim = int(vector_dim)
    embedding_index_name = f"{table_name}_embedding_hnsw_idx"
    doc_key_index_name = f"{table_name}_doc_key_uq"

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        if rebuild:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))

        cur.execute(
            sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS {{}} (
                    id SERIAL PRIMARY KEY,
                    doc_key TEXT,
                    content TEXT,
                    embedding VECTOR({vector_dim}),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            ).format(sql.Identifier(table_name))
        )
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS doc_key TEXT;").format(
                sql.Identifier(table_name)
            )
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops);"
            ).format(sql.Identifier(embedding_index_name), sql.Identifier(table_name))
        )
        cur.execute(
            sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {} (doc_key);").format(
                sql.Identifier(doc_key_index_name), sql.Identifier(table_name)
            )
        )
    conn.commit()
    mode_label = "rebuild" if rebuild else "upsert"
    print(f"ensured table '{table_name}' with vector({vector_dim}) [mode={mode_label}]")

def load_data(file_path: str) -> List[Dict]:
    """Loads JSONL data."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def main():
    parser = argparse.ArgumentParser(description="Finalize RAPTOR ingestion into PostgreSQL.")
    parser.add_argument(
        "--mode",
        choices=["upsert", "rebuild"],
        default="upsert",
        help="upsert preserves table and updates by doc_key; rebuild drops and recreates table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate DB/table setup only. Skip embedding, inserts, and BM25 build.",
    )
    args = parser.parse_args()

    print(f"Finalize mode: {args.mode}")
    print(f"Dry run: {args.dry_run}")
    if args.mode == "upsert":
        print("Note: On first deployment, run once with --mode rebuild for a clean initial baseline.")

    # 0. Optimization: Set PyTorch Env
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = load_config()
    db_table = config['database']['table_name']
    configured_device = config.get("embedding", {}).get("device", "auto")
    device = resolve_torch_device(configured_device)

    # Connect & Init DB
    try:
        conn = get_db_connection(config)
        ensure_table(conn, db_table, vector_dim=1024, rebuild=(args.mode == "rebuild"))
    except Exception as e:
        print(f"Database Error: {e}")
        print("Ensure PostgreSQL is running with pgvector extension.")
        sys.exit(1)

    if args.dry_run:
        print("Dry-run requested. Skipping embedding generation, DB upsert, and BM25 build.")
        conn.close()
        return

    # Unload Ollama to free VRAM
    print("Unloading any resident Ollama models to free VRAM...")
    try:
        # Support new config structure
        llm_cfg = config.get('llm_ingestion', config.get('llm', {}))
        model_name = llm_cfg.get('model_name')
        if model_name:
            LLMUtility.unload_model("ingestion")
        clear_torch_cache()
        import torch

        if device == "cuda" and torch.cuda.is_available():
            print(f"Cleared CUDA Cache. Free memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
        elif device == "mps":
            print("Cleared MPS Cache.")

    except Exception as e:
        print(f"Warning: Could not unload model: {e}")

    # Initialize Embeddings
    print(
        f"Initializing Embeddings (BAAI/bge-m3) on device: {device} "
        f"(configured: {configured_device} -> resolved: {device})..."
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=config['embedding']['model_name'],
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': config['embedding']['normalize_embeddings']}
    )

    # Load Data
    print("Loading Data...")
    l0_data = load_data(os.path.join("data", "raptor", "matrix_raptor_L0_data.jsonl"))
    l1_data = load_data(os.path.join("data", "raptor", "matrix_raptor_L1_data.jsonl"))
    
    all_data = l0_data + l1_data
    print(f"Total entries to process: {len(all_data)} (L0: {len(l0_data)}, L1: {len(l1_data)})")
    
    # Processing & Insertion
    batch_size = 8
    total_embedded = 0
    
    print("Starting Ingestion...")
    with conn.cursor() as cur:
        batch_texts = []
        batch_metas = []
        
        for i, item in enumerate(all_data):
            # Prepare Text Content
            text = item.get("summary", "")
            if not text:
                continue
            
            batch_texts.append(text)
            batch_metas.append(item)
            
            # Flush Batch
            if len(batch_texts) >= batch_size:
                # Embed
                vectors = embeddings.embed_documents(batch_texts)
                
                # Insert
                args_list = [
                    (build_doc_key(meta), text, vectors[j], Json(meta))
                    for j, (text, meta) in enumerate(zip(batch_texts, batch_metas))
                ]
                
                # Psycopg2 execute_values equivalent loop
                for doc_key_val, content_val, vector_val, meta_val in args_list:
                    cur.execute(
                        f"""
                        INSERT INTO {db_table} (doc_key, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (doc_key) DO UPDATE
                        SET content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            created_at = NOW()
                        """,
                        (doc_key_val, content_val, vector_val, meta_val)
                    )
                
                conn.commit()
                total_embedded += len(batch_texts)
                print(f"  Processed {total_embedded}/{len(all_data)}...")
                
                # Reset Batch
                batch_texts = []
                batch_metas = []
        
        # Flush Remaining
        if batch_texts:
            vectors = embeddings.embed_documents(batch_texts)
            args_list = [
                (build_doc_key(meta), text, vectors[j], Json(meta))
                for j, (text, meta) in enumerate(zip(batch_texts, batch_metas))
            ]
            for doc_key_val, content_val, vector_val, meta_val in args_list:
                cur.execute(
                    f"""
                    INSERT INTO {db_table} (doc_key, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_key) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                    """,
                    (doc_key_val, content_val, vector_val, meta_val)
                )
            conn.commit()
            total_embedded += len(batch_texts)
            
    print(f"Ingestion Complete. {total_embedded} vectors stored in table '{db_table}'.")
    
    # Build BM25 Index
    print("\nBuilding BM25 index for hybrid search...")
    try:
        from src.retrieve.bm25_search import BM25Index
        index_path = config.get("retrieval", {}).get("hybrid_search", {}).get("index_path", "data/bm25_index.pkl")
        bm25 = BM25Index(index_path)
        bm25.build_from_db(conn, db_table)
    except Exception as e:
        print(f"Warning: BM25 index build failed: {e}")
    
    conn.close()

if __name__ == "__main__":
    main()
