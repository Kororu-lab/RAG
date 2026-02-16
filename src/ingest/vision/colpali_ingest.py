import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from colpali_engine.models import ColPali, ColPaliProcessor
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import load_config, resolve_torch_device

VISION_DIR = os.path.join(os.path.dirname(__file__), "../../../data/ltdb_vision")
METADATA_FILE = os.path.join(VISION_DIR, "metadata.jsonl") 
COLPALI_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../../data/ltdb_colpali")
EMBEDDINGS_FILE = os.path.join(COLPALI_OUTPUT_DIR, "embeddings.pt")
METADATA_OUTPUT_FILE = os.path.join(COLPALI_OUTPUT_DIR, "indexed_metadata.json")

MODEL_NAME = "vidore/colpali-v1.2"
BATCH_SIZE = 4 

def load_metadata(path: str) -> List[Dict]:
    data = []
    if not os.path.exists(path):
        return data
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

    return data
    


def main():
    config = load_config()
    configured_device = config.get("embedding", {}).get("device", "auto")
    device = resolve_torch_device(configured_device)

    print(f"Using device: {device}")
    
    # MPS: Use float16 for best performance and memory usage (fp32 causes swapping on <32GB RAM)
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = ColPali.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    
    metadata = load_metadata(METADATA_FILE)
    print(f"Found {len(metadata)} entries.")
    
    if not metadata:
        return

    # Load and Preprocess Images (Generator)
    CHUNK_SIZE = 32 # Chunk 32
    all_embeddings = []
    final_metadata = []
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    for i in tqdm(range(0, len(metadata), CHUNK_SIZE)):
        chunk_meta = metadata[i : i + CHUNK_SIZE]
        chunk_images = []
        chunk_valid_meta = []
        
        for entry in chunk_meta:
            img_rel_path = entry.get("image_path")
            img_path = os.path.join(project_root, img_rel_path)
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    chunk_images.append(img)
                    chunk_valid_meta.append(entry)
                except Exception:
                    pass
        
        if not chunk_images:
            continue
        
        with torch.no_grad():
            # Process sub-batches within chunk
            for j in range(0, len(chunk_images), BATCH_SIZE):
                sub_imgs = chunk_images[j : j + BATCH_SIZE]
                batch_dict = processor.process_images(sub_imgs).to(model.device)
                batch_embeddings = model(**batch_dict)
                all_embeddings.append(batch_embeddings.cpu())

        final_metadata.extend(chunk_valid_meta)

    if not all_embeddings:
        print("No embeddings generated.")
        return

    # Concatenate all CPU tensors
    # List of [B, N, D]
    try:
        final_tensor = torch.cat(all_embeddings, dim=0)
    except:
        final_tensor = all_embeddings

    os.makedirs(COLPALI_OUTPUT_DIR, exist_ok=True)
    torch.save(final_tensor, EMBEDDINGS_FILE)
    
    with open(METADATA_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(final_metadata)} items to {COLPALI_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
