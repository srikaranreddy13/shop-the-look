"""
Step 3 (improved) - Evaluate with multi-crop scene matching.

A scene is a full outfit photo; the whole-image embedding is dominated by the
background and person. We instead split each scene into overlapping regions
(full + center + quadrants), embed each crop, and score every catalog product
by its BEST-matching crop. This compares garment-region to product instead of
whole-room to product, which lifts retrieval accuracy.

Metrics: Top-1 / Top-5 / Top-10 accuracy + MRR on validation pairs.
Outputs: prints metrics, writes metrics.json
"""
import json, os
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

IMG_SCENES = "images/scenes"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model = model.to(device).eval()

cat_emb = np.load("catalog_embeddings.npy")            # N x D, normalized
cat_ids = json.load(open("catalog_ids.json"))
id_to_row = {pid: i for i, pid in enumerate(cat_ids)}
cat_set = set(cat_ids)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def crops(img):
    """Full image + center + 4 quadrants (each slightly overlapping)."""
    w, h = img.size
    regions = [img]                                   # full
    cx0, cy0, cx1, cy1 = int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)
    regions.append(img.crop((cx0, cy0, cx1, cy1)))    # center
    mx, my = int(w*0.55), int(h*0.55)
    regions.append(img.crop((0, 0, mx, my)))          # top-left
    regions.append(img.crop((w-mx, 0, w, my)))        # top-right
    regions.append(img.crop((0, h-my, mx, h)))        # bottom-left
    regions.append(img.crop((w-mx, h-my, w, h)))      # bottom-right
    return regions


def embed_scene(sid):
    path = os.path.join(IMG_SCENES, sid + ".jpg")
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    with torch.no_grad():
        batch = torch.stack([preprocess(c) for c in crops(img)]).to(device)
        f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()                              # (num_crops x D)


val = read_jsonl("dataset/validation.jsonl")
ranks = []
scored = 0
for r in tqdm(val):
    prod, scene = r["product"], r["scene"]
    if prod not in cat_set:
        continue
    q = embed_scene(scene)                             # crops x D
    if q is None:
        continue
    # best crop similarity per catalog product
    sims = (cat_emb @ q.T).max(axis=1)                 # N
    order = np.argsort(-sims)
    true_row = id_to_row[prod]
    rank = int(np.where(order == true_row)[0][0]) + 1
    ranks.append(rank)
    scored += 1

ranks = np.array(ranks)
metrics = {
    "model": f"open_clip {MODEL_NAME} / {PRETRAINED} + multi-crop scene matching",
    "catalog_size": len(cat_ids),
    "pairs_scored": scored,
    "top1": float(np.mean(ranks <= 1)),
    "top5": float(np.mean(ranks <= 5)),
    "top10": float(np.mean(ranks <= 10)),
    "mrr": float(np.mean(1.0 / ranks)),
}

print(json.dumps(metrics, indent=2))
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics.json")