
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import json
import argparse
import psycopg2
import time
import torch
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import difflib

from src.llm.factory import get_llm
from src.utils import LLMUtility, load_config

try:
    from src.retrieve.colpali_search import ColPaliRetriever
except ImportError as e:
    print(f"Error importing ColPaliRetriever: {e}")
    ColPaliRetriever = None

try:
    from src.retrieve.bm25_search import BM25Index, reciprocal_rank_fusion
    BM25_AVAILABLE = True
except ImportError as e:
    print(f"BM25 not available: {e}")
    BM25_AVAILABLE = False



def extract_query_metadata(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts metadata (Language, Topic) from the user query.
    1. Language Detection (Fast Path or LLM)
    2. Topic Fallback (if Language is None)
    
    Returns: {'lang': str|None, 'topic': str|None}
    """
    metadata = {'lang': None, 'topic': None}
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Language Detection
    
    # Fast Path: Check against known directory names
    data_path = config["project"]["data_path"]
    doc_dir = os.path.join(data_path, "doc")
    known_languages = []
    if os.path.exists(doc_dir):
        known_languages = [d for d in os.listdir(doc_dir) if os.path.isdir(os.path.join(doc_dir, d))]
        
    query_lower = query.lower()
    
    # Check Fast Path
    for lang in known_languages:
        if lang in query_lower:
            metadata['lang'] = lang
            break

    # LLM fallback
    llm = None
    if not metadata['lang']:
        print("Simple detection failed. Using LLM for language detection...")
        llm_name = config.get("llm_retrieval", config.get("llm", {})).get("model_name")
        
        try:
            llm = get_llm("retrieval")
            
            lang_prompt = ChatPromptTemplate.from_template(
                """Analyze the user's query and identify if a specific language is being asked about.
If found, translate it to its standard English Academic name.
If no specific language is targeted, return "None".

- Query: "Language A의 성조는?" -> "Language A"
- Query: "What is the tone in Language B?" -> "Language B"
- Query: "Explain the grammar." -> "None"

Query: {query}
Language Name (English only, no punctuation):"""
            )
            
            chain = lang_prompt | llm | StrOutputParser()
            result = chain.invoke({"query": query}).strip()
            print(f"LLM Detected Language: {result}")
            
            if result and result != "None":
                clean_result = result.lower()
                if clean_result in known_languages:
                    metadata['lang'] = clean_result
                else:
                    matches = difflib.get_close_matches(clean_result, known_languages, n=1, cutoff=0.6)
                    if matches:
                        print(f"Mapped '{result}' to known language '{matches[0]}'")
                        metadata['lang'] = matches[0]
            
            # Topic fallback if language not detected
            if not metadata['lang']:
                print("Language not detected. Triggering Topic Fallback...")
                
                topic_prompt = ChatPromptTemplate.from_template(
                    """Analyze the user's query and classify it into one of the following Linguistic Domains:
- Phonology (Sounds, Tones, IPA, Phonemes)
- Grammar (Syntax, Morphology, Sentence Structure)
- Lexicon (Words, Vocabulary, Dictionary)
- General (History, Demographics, Metadata, Others)

Query: {query}
Domain (Return ONLY the category name):"""
                )
                
                chain_topic = topic_prompt | llm | StrOutputParser()
                topic_result = chain_topic.invoke({"query": query}).strip()
                
                # Basic validation
                valid_topics = ["Phonology", "Grammar", "Lexicon", "General"]
                if any(t in topic_result for t in valid_topics):
                     # Clean up partial matches
                     for t in valid_topics:
                         if t in topic_result:
                             metadata['topic'] = t
                             break
                else:
                    metadata['topic'] = "General"
                    
                print(f"LLM Detected Topic: {metadata['topic']}")
                
        except Exception as e:
            print(f"Metadata extraction failed: {e}")

    # Cleanup
    print("Ensuring VRAM is clean (Unloading Ollama)...")
    if llm: 
        del llm
    try:
        LLMUtility.unload_model("retrieval")
    except Exception:
        pass 
        
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    time.sleep(1) 
    
    return metadata


class RAGRetriever:
    def __init__(self):
        self.config = load_config()
        self.db_cfg = self.config.get("database", {})
        
        self.embedding_model_name = self.config["embedding"]["model_name"]
        self.device = self.config["embedding"]["device"]
        
        self.reranker_model_name = self.config["reranker"]["model_name"]
        self.use_reranker = self.config["reranker"]["enabled"]
        self.top_n = self.config["reranker"]["top_n"]
        
        # Connect to DB (Lightweight, keep open)
        print("Connecting to Vector Database (PostgreSQL)...")
        self.conn = psycopg2.connect(
            host=self.db_cfg.get('host', 'localhost'),
            port=self.db_cfg.get('port', 5432),
            user=self.db_cfg.get('user', 'user'),
            password=self.db_cfg.get('password', 'password'),
            dbname=self.db_cfg.get('dbname', 'ltdb_rag')
        )
        self.table_name = self.db_cfg.get('table_name', 'linguistics_raptor')
        
        # Hybrid search config
        hybrid_cfg = self.config.get("retrieval", {}).get("hybrid_search", {})
        self.hybrid_enabled = hybrid_cfg.get("enabled", False) and BM25_AVAILABLE
        self.vector_weight = hybrid_cfg.get("vector_weight", 0.6)
        self.bm25_weight = hybrid_cfg.get("bm25_weight", 0.4)
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)
        self.bm25_index_path = hybrid_cfg.get("index_path", "data/bm25_index.pkl")
        
        if self.hybrid_enabled:
            print(f"Hybrid Search: enabled (vector={self.vector_weight}, bm25={self.bm25_weight})")
        
        self.vision_keywords = ["도표", "table", "chart", "structure", "구조", "IPA", "paradigm", "gloss", "마커", "marker", "예시", "box", "박스"]

    def _clean_memory(self):
        """Force garbage collection and CUDA cache clearing."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_visual_query(self, query: str) -> bool:
        if not ColPaliRetriever:
            return False
        return any(k in query.lower() for k in self.vision_keywords)

    def retrieve_text(self, query: str, k: int, metadata: Dict[str, Any] = None):
        print(f"Loading Embeddings ({self.embedding_model_name})...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': self.config["embedding"]["normalize_embeddings"]}
        )
        
        query_vector = embeddings.embed_query(query)
        
        del embeddings
        self._clean_memory()
        print("Embeddings unloaded.")

        threshold_dist = 0.55
        fetch_k = k * 3
        
        # Build language filter clause
        lang_filter = ""
        lang_param = []
        if metadata and metadata.get('lang'):
            lang = metadata['lang']
            print(f"Applying Language Filter: {lang}")
            lang_filter = " AND metadata->>'lang' = %s"
            lang_param = [lang]
        
        # 1. Vector Search
        vector_sql = f"""
            SELECT id, content, metadata, (embedding <=> %s::vector) as distance
            FROM {self.table_name}
            WHERE (embedding <=> %s::vector) < %s {lang_filter}
            ORDER BY distance ASC LIMIT %s
        """
        vector_params = [query_vector, query_vector, threshold_dist] + lang_param + [fetch_k]
        
        vector_results = []  # [(id, distance)]
        doc_cache = {}       # id -> (content, metadata)
        
        with self.conn.cursor() as cur:
            cur.execute(vector_sql, tuple(vector_params))
            for row in cur.fetchall():
                doc_id, content, meta, dist = row
                vector_results.append((doc_id, 1 - dist))  # Convert to similarity
                doc_cache[doc_id] = (content, meta)
        
        print(f"Vector Search: {len(vector_results)} matches")
        
        # 2. BM25 Search (if enabled)
        bm25_results = []
        if self.hybrid_enabled:
            try:
                bm25_index = BM25Index(self.bm25_index_path)
                bm25_results = bm25_index.search(query, top_k=fetch_k)
                print(f"BM25 Search: {len(bm25_results)} matches")
            except Exception as e:
                print(f"BM25 search failed: {e}")
        
        # 3. Merge results
        if self.hybrid_enabled and bm25_results:
            # RRF Fusion
            rrf_scores = reciprocal_rank_fusion(
                vector_results, bm25_results,
                k=self.rrf_k,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight
            )
            
            # Sort by RRF score
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            
            # Fetch any docs from BM25 not in cache
            missing_ids = [doc_id for doc_id in sorted_ids if doc_id not in doc_cache]
            if missing_ids:
                placeholders = ','.join(['%s'] * len(missing_ids))
                with self.conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT id, content, metadata FROM {self.table_name}
                        WHERE id IN ({placeholders})
                    """, tuple(missing_ids))
                    for row in cur.fetchall():
                        doc_id, content, meta = row
                        doc_cache[doc_id] = (content, meta)
            
            # Build final doc list
            text_docs = []
            for doc_id in sorted_ids[:fetch_k]:
                if doc_id not in doc_cache:
                    continue
                content, meta = doc_cache[doc_id]
                actual_content = meta.get('original_content', content)
                doc = Document(page_content=actual_content, metadata=meta)
                doc.metadata['score'] = rrf_scores[doc_id]
                doc.metadata['_db_id'] = doc_id
                text_docs.append(doc)
            
            print(f"Hybrid Search (RRF): {len(text_docs)} merged results")
        else:
            # Vector-only fallback
            text_docs = []
            for doc_id, score in vector_results:
                content, meta = doc_cache[doc_id]
                actual_content = meta.get('original_content', content)
                doc = Document(page_content=actual_content, metadata=meta)
                doc.metadata['score'] = score
                doc.metadata['_db_id'] = doc_id
                text_docs.append(doc)
            
            print(f"Vector Search: {len(text_docs)} results (hybrid disabled)")
        
        return self._recursive_expand(text_docs)

    def _recursive_expand(self, docs: List[Document]) -> List[Document]:
        """
        [Recursive Retrieval Strategy]
        If a retrieved doc is Level 1 (Summary), expand it to its child Level 0 (Raw) chunks.
        This ensures the LLM sees the original detailed text, not just the summary.
        """
        expanded_docs = []
        l1_found = False
        
        # Collect all L1 child references
        # Set of (source_file, chunk_id)
        child_refs = set()
        
        for doc in docs:
            level = str(doc.metadata.get('level', '0'))
            
            if level == '0':
                expanded_docs.append(doc)
            elif level == '1':
                l1_found = True
                # Parse child_chunks. 
                # Old Schema: [int, int] (Legacy, can't recursive expand reliably without file path)
                # New Schema: [{"source_file": "...", "chunk_id": 0}, ...]
                children = doc.metadata.get('child_chunks', [])
                
                # Check schema by looking at first element
                if children and isinstance(children[0], dict):
                    for child in children:
                        src = child.get("source_file")
                        cid = child.get("chunk_id")
                        if src is not None and cid is not None:
                            child_refs.add((src, int(cid)))
                else:
                    # Legacy schema or empty. Keep L1 node as fallback.
                    # print("Warning: L1 node has legacy schema. Keeping summary.")
                    expanded_docs.append(doc)

        if not child_refs:
            return expanded_docs
            
        print(f"Recursive Retrieval: Expanding {len(child_refs)} L0 chunks from L1 nodes...")
        
        docs_l0 = []
        with self.conn.cursor() as cur:
            from collections import defaultdict
            file_map = defaultdict(list)
            for src, cid in child_refs:
                file_map[src].append(cid)
            
            or_clauses = []
            params = []
            for src, cids in file_map.items():
                placeholders = ",".join(["%s"] * len(cids))
                or_clauses.append(f"(metadata->>'source_file' = %s AND (metadata->>'chunk_id')::int IN ({placeholders}))")
                params.append(src)
                params.extend(cids)
            
            if or_clauses:
                where_clause = " OR ".join(or_clauses)
                sql = f"SELECT content, metadata FROM {self.table_name} WHERE ({where_clause})"
                
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
                
                for row in rows:
                    content, meta = row
                    # Verify level 0
                    if str(meta.get('level')) == '0':
                         # Use original_content if available, fallback to summary
                         actual_content = meta.get('original_content', content)
                         doc = Document(page_content=actual_content, metadata=meta)
                         # Inherit score? No, this is raw retrieval. 
                         # We will let Reranker handle scoring.
                         docs_l0.append(doc)
        
        # Deduplicate Expanded Docs vs Already Retrieved L0
        # Use (source_file, chunk_id) as key
        existing_keys = set()
        for d in expanded_docs:
            k = (d.metadata.get('source_file'), d.metadata.get('chunk_id'))
            existing_keys.add(k)
            
        final_docs = expanded_docs
        added_count = 0
        
        for d in docs_l0:
            k = (d.metadata.get('source_file'), d.metadata.get('chunk_id'))
            if k not in existing_keys:
                final_docs.append(d)
                existing_keys.add(k)
                added_count += 1
                
        print(f"Recursive Retrieval: Added {added_count} unique L0 chunks. Total Context: {len(final_docs)}")
        return final_docs

    def retrieve_vision(self, query: str, metadata: Dict[str, Any]):
        print("Initializing Vision Retriever (ColPali)...")
        vision_retriever = ColPaliRetriever()
        vision_retriever.initialize()
        
        # Use lang from metadata as filter
        lang_filter = metadata.get('lang')
        results = vision_retriever.search(query, top_k=3, lang_filter=lang_filter)
        
        # Cleanup Vision Model IMMEDIATELY
        del vision_retriever
        self._clean_memory()
        print("Vision Retriever unloaded.")
        
        return results

    def perform_rerank(self, query: str, docs: List[Document], top_n: int = None):
        if top_n is None:
            top_n = self.top_n
            
        print(f"Loading Reranker ({self.reranker_model_name})...")
        reranker = CrossEncoder(
            self.reranker_model_name, 
            device=self.device,
            trust_remote_code=True
        )
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs, batch_size=4) 
        
        del reranker
        self._clean_memory()
        print("Reranker unloaded.")
        
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_n_docs = [doc for doc, score in scored_docs[:top_n]]
        print(f"Reranked: Reduced from {len(docs)} to {len(top_n_docs)} docs (Top-N={top_n})")
        return top_n_docs

    def retrieve_documents(self, query: str, metadata: Dict[str, Any] = None, k: int = None, top_n: int = None) -> List[Document]:
        """
        Retrieves and returns a list of LangChain Document objects (Text + Vision).
        This method is designed for Agentic use (programmatic access).
        """
        print(f"Retrieving for query: {query}")
        
        # Determine K
        if k is None:
            k = self.config["retrieval"]["k"]
        print(f"Dynamic Top-K: {k}")
        
        # Determine Top-N
        if top_n is None:
            top_n = self.config["reranker"]["top_n"]
        print(f"Dynamic Rerank Top-N: {top_n}")

        if metadata:
            print(f"Metadata Context: {metadata}")
        else:
            metadata = {}

        # 1. Text Retrieval (Sequential)
        text_docs = self.retrieve_text(query, k, metadata=metadata)
        print(f"Text Matches: {len(text_docs)}")

        # 2. Vision Retrieval (Sequential)
        vision_docs = []
        if self._is_visual_query(query):
            print(f"Triggering Vision Search (ColPali)... Metadata: {metadata}")
            visual_results = self.retrieve_vision(query, metadata)
            print(f"Vision Search Matches: {len(visual_results)}")
            
            for res in visual_results:
                meta = res['metadata']
                # Create a pseudo-Document for Vision
                content = f"[Vision Content] Type: {meta.get('element_type')}\n"
                if 'context_preview' in meta:
                    content += f"Text in Image: {meta['context_preview']}\n"
                
                # Add score to metadata
                meta['score'] = res['score']
                meta['type'] = 'vision'
                
                doc = Document(page_content=content, metadata=meta)
                vision_docs.append(doc)

        # 3. Reranking (Sequential)
        if self.use_reranker and text_docs:
            print("Reranking text docs...")
            text_docs = self.perform_rerank(query, text_docs, top_n=top_n)

        # Merge Text and Vision
        final_docs = text_docs + vision_docs
        
        # Robust Reference ID Generation (Mutates metadata in place)
        for doc in final_docs:
             # Skip if already generated (e.g. Vision might have it)
             if 'ref_id' in doc.metadata: continue

             if doc.metadata.get('type') == 'vision':
                 doc.metadata['ref_id'] = f"{os.path.basename(doc.metadata.get('parent_filename','?'))} (Vision)"
                 continue

             ref_id = doc.metadata.get('group_key')
             if not ref_id:
                # Fallback for L0: file_basename:chunk_id (Header)
                src_file = doc.metadata.get('source_file', 'Unknown')
                src_base = os.path.basename(src_file)
                lang = doc.metadata.get('lang')
                
                if lang and lang.lower() not in ['unknown', 'none']:
                    display_name = f"{lang}/{src_base}"
                else:
                    display_name = src_base
                    
                chunk_id = doc.metadata.get('chunk_id', '?')
                header = doc.metadata.get('original_header', '')
                
                ref_id = f"{display_name}:{chunk_id}"
                if header:
                    ref_id += f" [{header}]"
             
             doc.metadata['ref_id'] = str(ref_id)

        return final_docs

    def retrieve(self, query: str, metadata: Dict[str, Any] = None):
        """
        Legacy wrapper for CLI usage. Saves context.json.
        """
        final_docs = self.retrieve_documents(query, metadata)
        
        # Format for JSON output
        text_context_lines = []
        vision_context_lines = []
        references = []
        
        seen = set()
        
        for doc in final_docs:
            # Context String Construction
            if doc.metadata.get('type') == 'vision':
                vision_context_lines.append(doc.page_content)
                
                # Reference
                ref_id = doc.metadata.get('ref_id')
                if ref_id not in seen:
                    references.append({
                        "type": "vision",
                        "element_type": doc.metadata.get('element_type'),
                        "score": doc.metadata.get('score'),
                        "file": doc.metadata.get('parent_filename'),
                        "image": doc.metadata.get('image_path')
                    })
                    seen.add(ref_id)
            else:
                # Text
                level = doc.metadata.get('level', '0')
                prefix = "[Summary]" if str(level) == '1' else "[Detail]"
                text_context_lines.append(f"{prefix} Source: {doc.metadata.get('ref_id', 'Chunk')}\n{doc.page_content}")
                
                # Reference
                ref_id = doc.metadata.get('ref_id')
                if ref_id not in seen:
                    references.append({
                        "type": "text",
                        "level": level,
                        "ref_id": ref_id,
                        "preview": doc.page_content[:50] + "..."
                    })
                    seen.add(ref_id)

        full_context = "\n\n".join(text_context_lines)
        if vision_context_lines:
             full_context += "\n\n[Visual Evidence Found]\n" + "\n---\n".join(vision_context_lines)

        payload = {
            "query": query,
            "metadata": metadata, 
            "context": full_context,
            "references": references
        }
        
        with open("context.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            
        print("Stage 1 Complete. Context saved to context.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User query")
    args = parser.parse_args()
    
    # Pre-Flight: Detect Metadata (Language/Topic)
    config = load_config()
    metadata = extract_query_metadata(args.query, config)
    
    # Initialize RAGRetriever (Loads Embeddings, Reranker, ColPali + Postgres)
    retriever = RAGRetriever()
    retriever.retrieve(args.query, metadata=metadata)
