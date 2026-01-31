import os
import sys
import json
import psycopg2
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from psycopg2.extras import Json

# Ensure src is resolvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils import load_config, LLMUtility 

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

def init_db(conn, table_name: str, vector_dim: int = 1024):
    """Initializes the database schema."""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Drop table if exists (for clean reload as requested)
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        
        # Create table with header_content and vector column
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR({vector_dim}),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # HNSW Index for fast approximate nearest neighbor search
        cur.execute(f"""
            CREATE INDEX ON {table_name} USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()
    print(f"initialized table '{table_name}' with vector({vector_dim})")

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
    # 0. Optimization: Set PyTorch Env
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    config = load_config()
    db_table = config['database']['table_name']
    
    # 0.1 Optimization: Force unload Ollama
    print("Unloading any resident Ollama models to free VRAM...")
    try:
        # Support new config structure
        llm_cfg = config.get('llm_ingestion', config.get('llm', {}))
        model_name = llm_cfg.get('model_name')
        if model_name:
            LLMUtility.unload_model("ingestion")
        import torch
        
        # Clear Cache based on Config Device or Availability
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Cleared CUDA Cache. Free memory: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("Cleared MPS Cache.")
            
    except Exception as e:
        print(f"Warning: Could not unload model: {e}")

    # 1. Initialize Embeddings
    device = config['embedding']['device']
    print(f"Initializing Embeddings (BAAI/bge-m3) on device: {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config['embedding']['model_name'],
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': config['embedding']['normalize_embeddings']}
    )
    
    # 2. Connect & Init DB
    try:
        conn = get_db_connection(config)
        init_db(conn, db_table, vector_dim=1024)
    except Exception as e:
        print(f"Database Error: {e}")
        print("Ensure PostgreSQL is running with pgvector extension.")
        sys.exit(1)
        
    # 3. Load Data
    print("Loading Data...")
    l0_data = load_data("matrix_raptor_L0_data.jsonl")
    l1_data = load_data("matrix_raptor_L1_data.jsonl")
    
    all_data = l0_data + l1_data
    print(f"Total entries to process: {len(all_data)} (L0: {len(l0_data)}, L1: {len(l1_data)})")
    
    # 4. Processing & Insertion
    batch_size = 8  # Reduced from 64 to prevent OOM
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
                    (text, vectors[j], Json(meta)) 
                    for j, (text, meta) in enumerate(zip(batch_texts, batch_metas))
                ]
                
                # Psycopg2 execute_values equivalent loop
                for content_val, vector_val, meta_val in args_list:
                    cur.execute(
                        f"INSERT INTO {db_table} (content, embedding, metadata) VALUES (%s, %s, %s)",
                        (content_val, vector_val, meta_val)
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
                (text, vectors[j], Json(meta)) 
                for j, (text, meta) in enumerate(zip(batch_texts, batch_metas))
            ]
            for content_val, vector_val, meta_val in args_list:
                cur.execute(
                    f"INSERT INTO {db_table} (content, embedding, metadata) VALUES (%s, %s, %s)",
                    (content_val, vector_val, meta_val)
                )
            conn.commit()
            total_embedded += len(batch_texts)
            
    conn.close()
    print(f"Ingestion Complete. {total_embedded} vectors stored in table '{db_table}'.")

if __name__ == "__main__":
    main()
