"""
Step 1 - Build a ~1000-product catalog subset and download all images.

Why a subset: the full catalog is ~50k products and each image is a Pinterest
download (slow, many 404s). We keep every validation product that exists in the
catalog (so accuracy stays measurable) and top up with random catalog products
to reach ~1000 distractors. We also download every validation scene image.

Outputs:
  images/catalog/<product_id>.jpg
  images/scenes/<scene_id>.jpg
  catalog_subset.json   (list of product ids that downloaded successfully)
"""
import json, os, random, time
import requests
from tqdm import tqdm

random.seed(42)

DATA = "dataset"
IMG_CATALOG = "images/catalog"
IMG_SCENES = "images/scenes"
TARGET_CATALOG_SIZE = 1000

os.makedirs(IMG_CATALOG, exist_ok=True)
os.makedirs(IMG_SCENES, exist_ok=True)


def convert_to_url(sig):
    prefix = "https://i.pinimg.com/400x/%s/%s/%s/%s.jpg"
    return prefix % (sig[0:2], sig[2:4], sig[4:6], sig)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download(sig, out_dir):
    out = os.path.join(out_dir, sig + ".jpg")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return True
    try:
        r = requests.get(convert_to_url(sig), timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            with open(out, "wb") as fh:
                fh.write(r.content)
            return True
    except Exception:
        pass
    return False


# --- load data ---
catalog_ids = [r["product"] for r in read_jsonl(f"{DATA}/product_catelog.jsonl")]
val = read_jsonl(f"{DATA}/validation.jsonl")

catalog_set = set(catalog_ids)
val_products = {r["product"] for r in val}
val_scenes = {r["scene"] for r in val}

# products from validation that actually exist in the catalog (must keep these)
must_keep = [p for p in val_products if p in catalog_set]

# top up to TARGET with random other catalog products as distractors
pool = [p for p in catalog_ids if p not in set(must_keep)]
random.shuffle(pool)
extra = pool[: max(0, TARGET_CATALOG_SIZE - len(must_keep))]
subset = list(dict.fromkeys(must_keep + extra))  # dedupe, preserve order

print(f"validation products in catalog : {len(must_keep)}")
print(f"distractors added              : {len(extra)}")
print(f"catalog subset target          : {len(subset)}")
print(f"unique validation scenes       : {len(val_scenes)}")

# --- download catalog images ---
print("\nDownloading catalog images...")
ok_catalog = []
for pid in tqdm(subset):
    if download(pid, IMG_CATALOG):
        ok_catalog.append(pid)
    time.sleep(0.02)

# --- download scene images ---
print("\nDownloading scene images...")
ok_scenes = []
for sid in tqdm(sorted(val_scenes)):
    if download(sid, IMG_SCENES):
        ok_scenes.append(sid)
    time.sleep(0.02)

with open("catalog_subset.json", "w") as f:
    json.dump(ok_catalog, f)

print(f"\nDownloaded catalog images : {len(ok_catalog)}/{len(subset)}")
print(f"Downloaded scene images   : {len(ok_scenes)}/{len(val_scenes)}")
print("Saved catalog_subset.json")
