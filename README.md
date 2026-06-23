---
title: Shop the Look
emoji: 🔍
colorFrom: yellow
colorTo: pink
sdk: streamlit
app_file: 04_demo.py
pinned: false
---

# Shop-the-Look: Visual Product Discovery

Visual retrieval system that finds catalog fashion products matching a scene
image. No text metadata exists in the dataset (catalog records are image IDs
only), so the approach is pure image-embedding retrieval.

## Approach
1. Download a ~1,000-product catalog subset (all validation products that exist
   in the catalog + random distractors) and all validation scene images.
2. Encode every catalog image with **Marqo-FashionSigLIP** (a fashion-tuned
   embedding model, see `config.py`) into a normalized embedding.
3. For a scene, embed it and rank catalog products by cosine similarity.
4. Evaluate with Top-1 / Top-5 / Top-10 accuracy and MRR on validation pairs.
5. Demo interface (Streamlit) shows top matches with similarity scores and a
   short color/category-based explanation per recommendation.

## Run in GitHub Codespaces
```bash
pip install -r requirements.txt
python 01_download.py        # downloads ~1k catalog + scene images (open internet needed)
python 02_build_index.py     # CLIP embeddings -> catalog_embeddings.npy
python 03_evaluate.py        # prints Top-K accuracy + MRR -> metrics.json
streamlit run 04_demo.py     # demo UI; click the forwarded-port popup
```

## Deploy

The model is set in one place (`config.py`) and shared by every script, so the
query encoder and catalog index can never drift apart.

### Hugging Face Spaces (recommended — free, 16 GB RAM)
The YAML header at the top of this README configures the Space (Streamlit SDK,
`app_file: 04_demo.py`). To deploy:
```bash
# 1. Create a new Space at huggingface.co/new-space  (SDK: Streamlit)
# 2. Push this repo to it:
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```
The Space builds from `requirements.txt`, runs `04_demo.py`, and serves the app.
`catalog_embeddings.npy` + images are committed, so no rebuild is needed at boot.

### Streamlit Community Cloud (free, ~1 GB RAM)
Point it at this repo with `04_demo.py` as the main file. FashionSigLIP is
B-sized so it fits the RAM cap (ViT-L-14 would not).

## Files
- `config.py` single source of truth for the embedding model
- `01_download.py` build subset + download images
- `02_build_index.py` build embedding index
- `03_evaluate.py` accuracy metrics
- `04_demo.py` Streamlit demo
- `dataset/` provided catalog + validation jsonl
