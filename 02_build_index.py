"""
Step 2 - Encode catalog images with a fashion-tuned embedding model.

Model: Marqo-FashionSigLIP (see config.py) -- fashion-specific, beats general
CLIP on garment retrieval while staying small enough for free hosting.
Embeddings are L2-normalized so a dot product equals cosine similarity.

Outputs:
  catalog_embeddings.npy   (N x D float32, normalized)
  catalog_ids.json         (product id per row, same order)
"""
import json, os
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from config import MODEL_REF, MODEL_LABEL

IMG_CATALOG = "images/catalog"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Model: {MODEL_LABEL}")

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_REF)
model = model.to(device).eval()

with open("catalog_subset.json") as f:
    ids = json.load(f)

embs, kept_ids = [], []
with torch.no_grad():
    for pid in tqdm(ids):
        path = os.path.join(IMG_CATALOG, pid + ".jpg")
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        x = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.cpu().numpy()[0])
        kept_ids.append(pid)

embs = np.stack(embs).astype("float32")
np.save("catalog_embeddings.npy", embs)
with open("catalog_ids.json", "w") as f:
    json.dump(kept_ids, f)

print(f"Encoded {len(kept_ids)} catalog products, dim={embs.shape[1]}")
print("Saved catalog_embeddings.npy and catalog_ids.json")