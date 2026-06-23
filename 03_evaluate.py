"""
Step 3 - Evaluate retrieval accuracy on the validation set.

For each validation (product, scene) pair we embed the scene, rank all catalog
products by cosine similarity, and check the rank of the true product. We report
the standard retrieval metrics your evaluator wants:

  Top-1  / Top-5 / Top-10 accuracy   (is the true product in the top K?)
  MRR                                 (mean reciprocal rank)

Only pairs whose true product is in our downloaded catalog AND whose scene image
downloaded are scored (others are not answerable and would unfairly penalize).

Outputs: prints metrics, writes metrics.json
"""
import json, os
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

IMG_SCENES = "images/scenes"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

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


def embed_scene(sid):
    path = os.path.join(IMG_SCENES, sid + ".jpg")
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


val = read_jsonl("dataset/validation.jsonl")
ranks = []
scored = 0
for r in tqdm(val):
    prod, scene = r["product"], r["scene"]
    if prod not in cat_set:
        continue
    q = embed_scene(scene)
    if q is None:
        continue
    sims = cat_emb @ q                       # cosine sims
    order = np.argsort(-sims)                # best first
    true_row = id_to_row[prod]
    rank = int(np.where(order == true_row)[0][0]) + 1   # 1-based
    ranks.append(rank)
    scored += 1

ranks = np.array(ranks)
metrics = {
    "model": f"open_clip {MODEL_NAME} / {PRETRAINED}",
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
