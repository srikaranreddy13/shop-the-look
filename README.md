# Shop-the-Look: Visual Product Discovery

Visual retrieval system that finds catalog fashion products matching a scene
image. No text metadata exists in the dataset (catalog records are image IDs
only), so the approach is pure image-embedding retrieval with CLIP.

## Approach
1. Download a ~1,000-product catalog subset (all validation products that exist
   in the catalog + random distractors) and all validation scene images.
2. Encode every catalog image with **open_clip ViT-B-32 (laion2b)** into a
   normalized embedding.
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

## Files
- `01_download.py` build subset + download images
- `02_build_index.py` build CLIP embedding index
- `03_evaluate.py` accuracy metrics
- `04_demo.py` Streamlit demo
- `dataset/` provided catalog + validation jsonl
