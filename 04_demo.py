"""
Shop-the-Look demo - premium redesign with live image upload.

Two input modes:
  - Upload: drop your own inspiration photo
  - Sample: pick a validation scene from the dataset

For any input the app embeds it with multi-crop CLIP (matching evaluation),
retrieves the top matching catalog products, and shows each with a similarity
score, a confidence cue, and a short explanation.

Run:  streamlit run 04_demo.py
"""
import io, base64, json, os
import numpy as np
import torch
import open_clip
from PIL import Image
import streamlit as st

IMG_SCENES = "images/scenes"
IMG_CATALOG = "images/catalog"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

COLORS = ["black", "white", "red", "blue", "green", "yellow", "pink",
          "brown", "grey", "beige", "navy", "purple", "orange", "cream"]
CATEGORIES = ["dress", "shirt", "t-shirt", "jacket", "coat", "sweater",
              "trousers", "jeans", "skirt", "shorts", "shoes", "bag", "hat"]


@st.cache_resource(show_spinner=False)
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model.eval(), preprocess, tokenizer


@st.cache_resource(show_spinner=False)
def load_index():
    return np.load("catalog_embeddings.npy"), json.load(open("catalog_ids.json"))


@st.cache_data(show_spinner=False)
def text_features(_model, _tokenizer, words):
    with torch.no_grad():
        toks = _tokenizer([f"a photo of a {w} clothing item" for w in words])
        tf = _model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf.numpy()


def crops(img):
    w, h = img.size
    out = [img, img.crop((int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)))]
    mx, my = int(w*0.55), int(h*0.55)
    out += [img.crop((0, 0, mx, my)), img.crop((w-mx, 0, w, my)),
            img.crop((0, h-my, mx, h)), img.crop((w-mx, h-my, w, h))]
    return out


def embed_scene(model, preprocess, img):
    with torch.no_grad():
        batch = torch.stack([preprocess(c) for c in crops(img)])
        f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy()


def top_tag(vec, tf, words):
    return words[int(np.argmax(tf @ vec))]


def confidence_label(score):
    if score >= 0.62:
        return "Strong match", "#3f7d58"
    if score >= 0.55:
        return "Good match", "#9a5b34"
    return "Loose match", "#a08a78"


def img_tag(pil_img, css_class):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img class="{css_class}" src="data:image/jpeg;base64,{b64}"/>'


# ---------------- page + styling ----------------
st.set_page_config(page_title="Shop the Look", page_icon="🔍", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');
.stApp { background:#faf8f5; }
h1,h2,h3 { font-family:'Fraunces',serif !important; color:#1a1714 !important; letter-spacing:-.01em; }
.block-container { padding-top:2.2rem; max-width:1200px; }
.eyebrow { font-family:'Inter',sans-serif; font-size:.72rem; letter-spacing:.22em;
           text-transform:uppercase; color:#9a8f80; margin-bottom:.3rem; }
.lede { font-family:'Inter',sans-serif; color:#6b6055; font-size:.95rem; margin-top:-.3rem; }
.rule { height:1px; background:#e7e0d6; border:0; margin:1.4rem 0; }
.score { font-family:'Fraunces',serif; font-size:1.1rem; color:#1a1714; margin-top:.5rem; }
.why { color:#6b6055; font-size:.82rem; line-height:1.35; margin-top:.1rem; }
.rank { font-family:'Inter'; font-size:.7rem; letter-spacing:.15em; color:#b3a899;
        text-transform:uppercase; margin-bottom:.35rem; }
.conf { font-family:'Inter'; font-size:.7rem; font-weight:600; margin-top:.25rem; }
.pct { font-family:'Inter'; font-size:.68rem; color:#9a8f80; margin-top:.15rem; }
.bar { height:4px; border-radius:3px; background:#ece5db; margin-top:.3rem; overflow:hidden; }
.bar > span { display:block; height:100%; border-radius:3px; }
.match-img { width:100%; height:230px; object-fit:cover; border-radius:8px;
             box-shadow:0 1px 3px rgba(26,23,20,.06); }
.scene-img { width:100%; border-radius:8px; box-shadow:0 1px 3px rgba(26,23,20,.06); }
section[data-testid="stSidebar"] { background:#f3eee7; border-right:1px solid #e7e0d6; }
section[data-testid="stSidebar"] * { font-family:'Inter',sans-serif; color:#1a1714 !important; }
section[data-testid="stSidebar"] .eyebrow { color:#9a8f80 !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#6b6055 !important; }
div[data-testid="stFileUploader"] section { border:1.5px dashed #cdc2b2; border-radius:10px; background:#fdfcfa; }
.empty { font-family:'Inter'; color:#9a8f80; text-align:center; padding:3rem 1rem;
         border:1.5px dashed #ddd4c6; border-radius:12px; background:#fdfcfa; }
.stTabs [data-baseweb="tab-list"] { gap:1.5rem; }
.stTabs [data-baseweb="tab"] { font-family:'Inter'; font-weight:500; }
</style>
""", unsafe_allow_html=True)

model, preprocess, tokenizer = load_model()
cat_emb, cat_ids = load_index()
color_tf = text_features(model, tokenizer, COLORS)
cat_tf = text_features(model, tokenizer, CATEGORIES)

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Settings</div>', unsafe_allow_html=True)
    topk = st.slider("Number of matches", 3, 8, 5)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
    st.caption(f"open_clip {MODEL_NAME}\n\nCatalog: {len(cat_ids)} products\n\n"
               "Retrieval: multi-crop cosine similarity")

# ---------------- header ----------------
st.markdown('<div class="eyebrow">Visual Product Discovery</div>', unsafe_allow_html=True)
st.markdown("# Shop the Look")
st.markdown('<p class="lede">Upload an inspiration photo or pick a sample scene to find '
            'the closest products in the catalog.</p>', unsafe_allow_html=True)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------- input ----------------
scene_img = None
tab_upload, tab_sample = st.tabs(["Upload image", "Sample scenes"])

with tab_upload:
    up = st.file_uploader("Drop a fashion photo (JPG or PNG)",
                          type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if up is not None:
        try:
            scene_img = Image.open(up).convert("RGB")
        except Exception:
            st.error("That file could not be read as an image. Try a JPG or PNG.")

with tab_sample:
    scenes = sorted(f[:-4] for f in os.listdir(IMG_SCENES) if f.endswith(".jpg"))
    labels = [f"Scene {i:02d}" for i in range(1, len(scenes) + 1)]
    pick = st.selectbox("Sample scene", labels, label_visibility="collapsed")
    if scene_img is None and pick:
        sid = scenes[labels.index(pick)]
        scene_img = Image.open(os.path.join(IMG_SCENES, sid + ".jpg")).convert("RGB")

# ---------------- results ----------------
if scene_img is None:
    st.markdown('<div class="empty">No image yet — upload a photo above '
                'or choose a sample scene to see matches.</div>', unsafe_allow_html=True)
else:
    with st.spinner("Finding the closest products…"):
        q = embed_scene(model, preprocess, scene_img)
        sims = (cat_emb @ q.T).max(axis=1)
        order = np.argsort(-sims)[:topk]
        scene_color = top_tag(q[0], color_tf, COLORS)
        scene_cat = top_tag(q[0], cat_tf, CATEGORIES)

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown('<div class="eyebrow">The Look</div>', unsafe_allow_html=True)
        st.markdown(img_tag(scene_img, "scene-img"), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="eyebrow">Closest Matches</div>', unsafe_allow_html=True)
        cols = st.columns(topk, gap="medium")
        for i, (c, idx) in enumerate(zip(cols, order), 1):
            pid = cat_ids[idx]
            score = float(sims[idx])
            pim = Image.open(os.path.join(IMG_CATALOG, pid + ".jpg")).convert("RGB")
            pcolor = top_tag(cat_emb[idx], color_tf, COLORS)
            pcat = top_tag(cat_emb[idx], cat_tf, CATEGORIES)
            shared = []
            if pcolor == scene_color: shared.append(f"shared {pcolor} tone")
            if pcat == scene_cat: shared.append(f"same item type ({pcat})")
            why = ", ".join(shared) if shared else f"closest visual match — a {pcolor} {pcat}"
            label, color = confidence_label(score)
            pct = int(min(max((score - 0.45) / 0.25, 0), 1) * 100)
            with c:
                st.markdown(f'<div class="rank">No. {i}</div>', unsafe_allow_html=True)
                st.markdown(img_tag(pim, "match-img"), unsafe_allow_html=True)
                st.markdown(
                    f'<div class="score">{score:.2f}</div>'
                    f'<div class="bar"><span style="width:{pct}%;background:{color}"></span></div>'
                    f'<div class="conf" style="color:{color}">{label}</div>'
                    f'<div class="pct">{pct}% confidence</div>'
                    f'<div class="why">{why.capitalize()}.</div>',
                    unsafe_allow_html=True)