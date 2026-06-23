"""
Step 4 - Demo interface (Shop-the-Look).

Pick a validation scene, the app embeds it with CLIP, retrieves the top-K most
similar catalog products, and shows each with a similarity score and a short
generated explanation (color + visual-appearance based, since the only signal
available is the image itself).

Run:  streamlit run 04_demo.py
Codespaces auto-forwards the port; click the popup to open in browser.
"""
import json, os
import numpy as np
import torch
import open_clip
from PIL import Image
import streamlit as st

IMG_SCENES = "images/scenes"
IMG_CATALOG = "images/catalog"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
TOPK = 5

# CLIP zero-shot tags used only to phrase explanations
COLORS = ["black", "white", "red", "blue", "green", "yellow", "pink",
          "brown", "grey", "beige", "navy", "purple", "orange"]
CATEGORIES = ["dress", "shirt", "t-shirt", "jacket", "coat", "sweater",
              "trousers", "jeans", "skirt", "shoes", "bag", "hat"]


@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model.eval(), preprocess, tokenizer


@st.cache_resource
def load_index():
    emb = np.load("catalog_embeddings.npy")
    ids = json.load(open("catalog_ids.json"))
    return emb, ids


@st.cache_data
def text_features(_model, _tokenizer, words):
    with torch.no_grad():
        toks = _tokenizer([f"a photo of a {w} clothing item" for w in words])
        tf = _model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf.numpy()


def embed_image(model, preprocess, img):
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy()[0]


def top_tag(vec, tf, words):
    return words[int(np.argmax(tf @ vec))]


model, preprocess, tokenizer = load_model()
cat_emb, cat_ids = load_index()
color_tf = text_features(model, tokenizer, COLORS)
cat_tf = text_features(model, tokenizer, CATEGORIES)

st.set_page_config(page_title="Shop-the-Look", layout="wide")
st.title("Shop-the-Look: Visual Product Discovery")
st.caption(f"Model: open_clip {MODEL_NAME} ({PRETRAINED}) · catalog {len(cat_ids)} products")

scenes = sorted(f[:-4] for f in os.listdir(IMG_SCENES) if f.endswith(".jpg"))
scene_id = st.selectbox("Choose a scene image", scenes)

if scene_id:
    scene_img = Image.open(os.path.join(IMG_SCENES, scene_id + ".jpg")).convert("RGB")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Scene")
        st.image(scene_img, use_container_width=True)

    q = embed_image(model, preprocess, scene_img)
    sims = cat_emb @ q
    order = np.argsort(-sims)[:TOPK]
    scene_color = top_tag(q, color_tf, COLORS)
    scene_cat = top_tag(q, cat_tf, CATEGORIES)

    with right:
        st.subheader("Top matches")
        cols = st.columns(TOPK)
        for c, idx in zip(cols, order):
            pid = cat_ids[idx]
            score = float(sims[idx])
            pim = Image.open(os.path.join(IMG_CATALOG, pid + ".jpg")).convert("RGB")
            pcolor = top_tag(cat_emb[idx], color_tf, COLORS)
            pcat = top_tag(cat_emb[idx], cat_tf, CATEGORIES)
            with c:
                st.image(pim, use_container_width=True)
                st.markdown(f"**{score:.2f}** similarity")
                shared = []
                if pcolor == scene_color:
                    shared.append(f"same **{pcolor}** tone")
                if pcat == scene_cat:
                    shared.append(f"both look like a **{pcat}**")
                why = "; ".join(shared) if shared else \
                    f"closest visual appearance (a {pcolor} {pcat})"
                st.caption(f"Why: {why}.")
