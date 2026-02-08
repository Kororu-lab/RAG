import os
import json
import torch
from typing import List, Dict
from colpali_engine.models import ColPali, ColPaliProcessor

COLPALI_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/ltdb_colpali")
EMBEDDINGS_FILE = os.path.join(COLPALI_OUTPUT_DIR, "embeddings.pt")
METADATA_OUTPUT_FILE = os.path.join(COLPALI_OUTPUT_DIR, "indexed_metadata.json")
MODEL_NAME = "vidore/colpali-v1.2"

class ColPaliRetriever:
    _instance = None
    _model = None
    _processor = None
    _embeddings = None
    _metadata = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ColPaliRetriever, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        if self._model is not None:
            return

        if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(METADATA_OUTPUT_FILE):
             print("ColPali Index not found. Vision search disabled.")
             return

        print("Loading ColPali Model...")
        # Load Config for Device
        import yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "..", "config.yaml")
        with open(config_path, "r") as f:
             config = yaml.safe_load(f)
        
        configured_device = config.get("embedding", {}).get("device", "auto")
        device = "cpu"
        if configured_device == "mps":
            if torch.backends.mps.is_available():
                device = "mps"
        elif configured_device == "cuda":
             if torch.cuda.is_available():
                 device = "cuda"
        else:
             if torch.cuda.is_available():
                 device = "cuda"
             elif torch.backends.mps.is_available():
                 device = "mps"
        
        # MPS: Use float16 for best performance and memory usage
        if device == "cuda":
            dtype = torch.bfloat16
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        self._model = ColPali.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, device_map=device
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
        
        print("Loading ColPali Index...")
        self._embeddings = torch.load(EMBEDDINGS_FILE, map_location="cpu")
        with open(METADATA_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            self._metadata = json.load(f)

    def search(self, query: str, top_k: int = 3, lang_filter: str = None) -> List[Dict]:
        if self._model is None:
            self.initialize()
            if self._model is None:
                return []

        if self._embeddings is None:
            return []

        # Ensure embeddings are tensor
        doc_embeddings = self._embeddings
        if isinstance(doc_embeddings, list):
            doc_embeddings = torch.stack(doc_embeddings)
            
        with torch.no_grad():
            batch_query = self._processor.process_queries([query]).to(self._model.device)
            query_embeddings = self._model(**batch_query)

        doc_embeddings = doc_embeddings.to(query_embeddings.device)
        Q = query_embeddings[0]

        # MaxSim
        scores = torch.einsum("bnd,qd->bnq", doc_embeddings, Q)
        max_scores = scores.max(dim=1).values
        total_scores = max_scores.sum(dim=1)
        
        # Sort all scores descending
        sorted_indices = torch.argsort(total_scores, descending=True)
        
        results = []
        for idx in sorted_indices:
            idx_val = idx.item()
            metadata = self._metadata[idx_val]
            
            # Metadata Filtering (supports single lang or list)
            if lang_filter:
                # Normalize to list
                langs = lang_filter if isinstance(lang_filter, list) else [lang_filter]
                parent_path = metadata.get("parent_path", "").lower()
                # Check if any of the languages match
                if not any(lang.lower() in parent_path for lang in langs):
                    continue
            
            results.append({
                "score": total_scores[idx_val].item(),
                "metadata": metadata
            })
            
            if len(results) >= top_k:
                break
            
        return results
