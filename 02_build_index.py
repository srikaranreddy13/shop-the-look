"""
Step 2 - Encode every downloaded catalog image into a CLIP embedding.

Model: open_clip ViT-B-32, pretrained on laion2b. This is the "model used"
your evaluator asks about. Embeddings are L2-normalized so a dot product equals
cosine similarity.

Outputs:
  catalog_embeddings.npy   (N x D float32, normalized)
  catalog_ids.json         (the product id for each row, same order)
"""
import json, os
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

IMG_CATALOG = "images/catalog"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
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
