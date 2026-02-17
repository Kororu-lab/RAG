import os
import json
import torch
from typing import List, Dict
from colpali_engine.models import ColPali, ColPaliProcessor
from src.utils import (
    load_config,
    resolve_torch_device,
    resolve_vision_device_and_dtype,
)

COLPALI_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/ltdb_colpali")
EMBEDDINGS_FILE = os.path.join(COLPALI_OUTPUT_DIR, "embeddings.pt")
METADATA_OUTPUT_FILE = os.path.join(COLPALI_OUTPUT_DIR, "indexed_metadata.json")
MODEL_NAME = "vidore/colpali-v1.2"


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ColPaliRetriever:
    _instance = None
    _model = None
    _processor = None
    _embeddings = None
    _metadata = None
    _device = "cpu"
    _dtype = torch.float32

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
        config = load_config()
        vision_cfg = config.get("vision", {}) or {}
        embedding_cfg = config.get("embedding", {}) or {}
        requested = vision_cfg.get("device") or embedding_cfg.get("device") or "auto"
        raw_resolved = resolve_torch_device(requested)
        device, dtype = resolve_vision_device_and_dtype(config)
        self._device = device
        self._dtype = dtype

        if (
            raw_resolved == "mps"
            and device == "cpu"
            and vision_cfg.get("mps_fallback_to_cpu", True)
        ):
            print(
                "[Vision] MPS detected but using CPU for ColPali stability "
                "(vision.mps_fallback_to_cpu=true)."
            )
        
        self._model = ColPali.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, device_map=device
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
        
        print("Loading ColPali Index...")
        self._embeddings = _safe_torch_load(EMBEDDINGS_FILE)
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
            batch_query = self._processor.process_queries([query]).to(self._device)
            query_embeddings = self._model(**batch_query)

        if isinstance(query_embeddings, (tuple, list)):
            query_embeddings = query_embeddings[0]

        query_embeddings = query_embeddings.to(device=self._device, dtype=self._dtype)
        doc_embeddings = doc_embeddings.to(device=self._device, dtype=self._dtype)
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
