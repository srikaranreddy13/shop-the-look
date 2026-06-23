"""
Single source of truth for the embedding model.

All scripts (build index, evaluate, demo) MUST import MODEL_REF from here so the
query encoder and the catalog encoder can never drift apart -- mismatched models
produce different embedding dimensions / spaces and silently break retrieval.

Marqo-FashionSigLIP is a SigLIP model fine-tuned for fashion product retrieval.
It outperforms general CLIP (incl. ViT-L-14) on this task while staying B-sized,
so it also fits free Streamlit / small hosts. Loaded straight from the Hub via
open_clip -- no separate `pretrained` tag is needed.
"""
MODEL_REF = "hf-hub:Marqo/marqo-fashionSigLIP"
MODEL_LABEL = "Marqo-FashionSigLIP (fashion-tuned)"
