"""AI Neural Style Transfer Studio — Streamlit UI.

All model logic lives in pipeline.py; this file is presentation, state and
orchestration only. Results are kept in st.session_state so they survive
reruns (e.g. clicking a download button).
"""
import base64
import os
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

import pipeline

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="AI Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.hero {
    text-align: center;
    padding: 2.2rem 1.5rem 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 55%, #b53fd1 100%);
    border-radius: 16px;
    color: #fff;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, .30);
}
.hero h1 {font-size: 2.15rem; margin: 0 0 .45rem; letter-spacing: .3px;}
.hero p {opacity: .93; margin: 0; font-size: 1.06rem;}
.hero .badge {
    display: inline-block; padding: .2rem .75rem; margin: .9rem .2rem 0;
    border-radius: 999px; background: rgba(255,255,255,.16);
    font-size: .8rem; backdrop-filter: blur(4px);
}
div[data-testid="stMetric"] {
    background: rgba(102, 126, 234, .07);
    border: 1px solid rgba(102, 126, 234, .25);
    border-radius: 12px;
    padding: .8rem .9rem;
}
div[data-testid="stImage"] img {border-radius: 10px;}
.stButton > button, .stDownloadButton > button {border-radius: 10px;}
.stFileUploader {border-radius: 12px;}
.footer-card {
    text-align: center; padding: 1.6rem; margin-top: 2rem;
    background: linear-gradient(90deg, rgba(102,126,234,.10), rgba(118,75,162,.10));
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎨 AI Neural Style Transfer Studio</h1>
    <p>Transform your photos into artwork with Adaptive Instance Normalization</p>
    <span class="badge">⚡ Real-time AdaIN</span>
    <span class="badge">🎭 Multi-style blending</span>
    <span class="badge">🌈 Color preservation</span>
    <span class="badge">🔍 Quality metrics</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------- models
APP_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Loading neural networks…")
def get_models():
    return pipeline.load_models(os.path.join(APP_DIR, "models"))


try:
    encoder, decoder, device = get_models()
except FileNotFoundError as e:
    st.error(f"❌ Model weights not found: `{e}`\n\n"
             "Place `vgg_normalised.pth` and `decoder.pth` in the `models/` folder.")
    st.stop()
except Exception as e:  # corrupt weights, torch issues, …
    st.error(f"❌ Failed to load neural network models: {e}")
    st.stop()

# ------------------------------------------------------- presets & defaults
DEFAULTS = {
    "alpha": 0.70, "guidance": 0.30, "guided": True, "passes": 1,
    "color_preserve": False, "enhance": True, "sharpen": 1.2,
    "contrast": 1.1, "saturation": 1.0, "noise_reduction": True,
    "proc_size": 768,
}

PRESETS = {
    "Artistic Strong":  {"alpha": 0.85, "guidance": 0.15, "guided": True,
                         "passes": 2, "color_preserve": False,
                         "sharpen": 1.35, "contrast": 1.15, "saturation": 1.10},
    "Artistic Subtle":  {"alpha": 0.55, "guidance": 0.40, "guided": True,
                         "passes": 1, "color_preserve": False,
                         "sharpen": 1.10, "contrast": 1.05, "saturation": 1.05},
    "Photographic":     {"alpha": 0.45, "guidance": 0.50, "guided": True,
                         "passes": 1, "color_preserve": True,
                         "sharpen": 1.00, "contrast": 1.00, "saturation": 0.95},
    "Abstract":         {"alpha": 0.95, "guidance": 0.05, "guided": False,
                         "passes": 3, "color_preserve": False,
                         "sharpen": 1.50, "contrast": 1.25, "saturation": 1.15},
    "Texture Heavy":    {"alpha": 0.80, "guidance": 0.20, "guided": True,
                         "passes": 2, "color_preserve": False,
                         "sharpen": 1.45, "contrast": 1.15, "saturation": 1.00},
    "Content Preserve": {"alpha": 0.35, "guidance": 0.65, "guided": True,
                         "passes": 1, "color_preserve": True,
                         "sharpen": 1.00, "contrast": 1.00, "saturation": 1.00},
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("history", [])


def _apply_preset():
    preset = st.session_state.get("preset", "Custom")
    for key, value in PRESETS.get(preset, {}).items():
        st.session_state[key] = value


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    icon = "🚀" if device.type == "cuda" else "💻"
    st.success(f"{icon} Running on **{device.type.upper()}**")

    st.markdown("## 🎨 Style Controls")
    st.selectbox("🎯 Preset", ["Custom"] + list(PRESETS.keys()),
                 key="preset", on_change=_apply_preset,
                 help="Presets prefill the controls below — tweak them freely afterwards")

    st.slider("Style strength (α)", 0.0, 1.0, step=0.05, key="alpha",
              help="Higher = more style, less content")
    st.slider("Content preservation", 0.0, 1.0, step=0.05, key="guidance",
              help="How strongly edges and structure of the original are protected")
    st.toggle("Guided transfer (content-aware)", key="guided",
              help="Protects structurally important regions (faces, edges, text)")
    st.slider("Style refinement passes", 1, 3, step=1, key="passes",
              help="Each pass re-stylizes the result — stronger, more painterly output")
    st.toggle("Preserve original colors", key="color_preserve",
              help="Apply the style's texture but keep your photo's color palette")

    st.markdown("## 🔧 Output & Post-Processing")
    st.select_slider("Processing resolution", options=[384, 512, 768, 1024],
                     key="proc_size",
                     help="Higher = finer detail but slower. Output is always "
                          "rendered back at your image's full resolution.")
    st.toggle("Quality enhancement", key="enhance")
    if st.session_state.enhance:
        st.slider("Sharpening", 0.5, 2.5, step=0.05, key="sharpen")
        st.slider("Contrast", 0.5, 2.0, step=0.05, key="contrast")
        st.slider("Saturation", 0.5, 2.0, step=0.05, key="saturation")
        st.toggle("Noise reduction", key="noise_reduction")

# ------------------------------------------------------------------ uploads
st.markdown("## 📤 Upload Your Images")
up_col1, up_col2 = st.columns(2, gap="large")

with up_col1:
    st.markdown("### 📷 Content Image")
    content_file = st.file_uploader(
        "The photo to transform", type=["jpg", "jpeg", "png", "webp"],
        help="This image's structure is kept; the style is painted onto it")
    content_img = None
    if content_file:
        content_img = ImageOps.exif_transpose(
            Image.open(content_file)).convert("RGB")
        st.image(content_img, caption=f"{content_img.width}×{content_img.height}px",
                 use_container_width=True)

with up_col2:
    st.markdown("### 🎭 Style Images")
    style_files = st.file_uploader(
        "One or more artworks to borrow the style from",
        type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    style_imgs = []
    if style_files:
        style_imgs = [ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                      for f in style_files]
        thumb_cols = st.columns(min(len(style_imgs), 4))
        for i, img in enumerate(style_imgs):
            with thumb_cols[i % len(thumb_cols)]:
                st.image(img, caption=f"Style {i + 1}", use_container_width=True)

# Style blend weights (only relevant for 2+ styles)
manual_weights = None
if len(style_imgs) > 1:
    st.markdown("#### ⚖️ Style Blending")
    mode = st.radio("How should the styles be mixed?",
                    ["Auto (content-aware)", "Manual weights"],
                    horizontal=True, label_visibility="collapsed")
    if mode == "Manual weights":
        weight_cols = st.columns(min(len(style_imgs), 4))
        manual_weights = []
        for i in range(len(style_imgs)):
            with weight_cols[i % len(weight_cols)]:
                manual_weights.append(
                    st.slider(f"Style {i + 1}", 0.0, 1.0, 1.0 / len(style_imgs),
                              0.05, key=f"style_w_{i}"))
    else:
        st.caption("💡 Weights are derived from each style's similarity to your content image.")

# ----------------------------------------------------------------- generate
st.markdown("---")
_, gen_col, _ = st.columns([1, 2, 1])
with gen_col:
    generate = st.button("🚀 Create Artistic Masterpiece", type="primary",
                         use_container_width=True)

if generate:
    if content_img is None or not style_imgs:
        st.error("🚫 Please upload a content image **and** at least one style image.")
    else:
        s = st.session_state
        started = time.perf_counter()
        with st.status("🎨 Creating your artwork…", expanded=True) as status:
            st.write(f"🧠 Stylizing at {s.proc_size}px "
                     f"({s.passes} pass{'es' if s.passes > 1 else ''}, α={s.alpha:.2f})…")
            out_img, used_weights = pipeline.stylize(
                encoder, decoder, content_img, style_imgs,
                alpha=s.alpha, processing_size=s.proc_size, passes=s.passes,
                guided=s.guided, guidance=s.guidance,
                style_weights=manual_weights, preserve_color=s.color_preserve,
                device=device)

            if s.enhance:
                st.write("✨ Applying quality enhancement…")
                out_img = Image.fromarray(pipeline.post_process(
                    np.asarray(out_img), sharpen=s.sharpen, contrast=s.contrast,
                    saturation=s.saturation, noise_reduction=s.noise_reduction))

            st.write("📊 Computing quality metrics…")
            metrics = pipeline.compute_metrics(content_img, out_img)

            # Pre-encode downloads once so reruns don't recompute them.
            buf_png, buf_jpg, buf_cmp = BytesIO(), BytesIO(), BytesIO()
            out_img.save(buf_png, "PNG")
            out_img.save(buf_jpg, "JPEG", quality=95, optimize=True)
            gap, h = 10, max(content_img.height, out_img.height)
            side_by_side = Image.new(
                "RGB", (content_img.width + out_img.width + gap, h), "white")
            side_by_side.paste(content_img, (0, (h - content_img.height) // 2))
            side_by_side.paste(out_img, (content_img.width + gap,
                                         (h - out_img.height) // 2))
            side_by_side.save(buf_cmp, "PNG")

            elapsed = time.perf_counter() - started
            status.update(label=f"✅ Masterpiece ready in {elapsed:.1f}s",
                          state="complete", expanded=False)

        preset = s.get("preset", "Custom")
        st.session_state.result = {
            "content": content_img, "styles": style_imgs, "output": out_img,
            "weights": used_weights, "metrics": metrics, "elapsed": elapsed,
            "png": buf_png.getvalue(), "jpg": buf_jpg.getvalue(),
            "cmp": buf_cmp.getvalue(),
            "summary": (f"{preset} · α {s.alpha:.2f} · {s.passes} pass"
                        f"{'es' if s.passes > 1 else ''} · {s.proc_size}px"
                        f"{' · colors preserved' if s.color_preserve else ''}"),
            "when": datetime.now().strftime("%H:%M:%S"),
        }
        thumb = out_img.copy()
        thumb.thumbnail((280, 280))
        st.session_state.history = (
            [{"thumb": thumb, "caption": f"{st.session_state.result['when']} · {preset}"}]
            + st.session_state.history)[:8]


# ------------------------------------------------------------------ results
def _b64_jpeg(img: Image.Image, max_side: int = 1080) -> str:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def comparison_slider(before: Image.Image, after: Image.Image, height: int = 520):
    """Interactive before/after slider, fully self-contained HTML."""
    b, a = _b64_jpeg(before), _b64_jpeg(after)
    inner = height - 10
    components.html(f"""
<div id="wrap" style="position:relative;width:100%;height:{inner}px;background:#0e1117;
        border-radius:14px;overflow:hidden;font-family:sans-serif;user-select:none">
  <img src="data:image/jpeg;base64,{a}"
       style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain">
  <img id="topimg" src="data:image/jpeg;base64,{b}"
       style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;
              clip-path:inset(0 50% 0 0)">
  <div id="line" style="position:absolute;top:0;bottom:0;left:50%;width:2px;
       background:#fff;box-shadow:0 0 8px rgba(0,0,0,.7);pointer-events:none"></div>
  <div id="knob" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
       width:36px;height:36px;border-radius:50%;background:#fff;color:#444;display:flex;
       align-items:center;justify-content:center;font-weight:700;font-size:15px;
       pointer-events:none;box-shadow:0 2px 10px rgba(0,0,0,.45)">⇔</div>
  <span style="position:absolute;top:12px;left:12px;background:rgba(0,0,0,.55);color:#fff;
        padding:3px 12px;border-radius:999px;font-size:12px">Original</span>
  <span style="position:absolute;top:12px;right:12px;background:rgba(0,0,0,.55);color:#fff;
        padding:3px 12px;border-radius:999px;font-size:12px">Stylized</span>
  <input id="rng" type="range" min="0" max="100" value="50"
         style="position:absolute;inset:0;width:100%;height:100%;opacity:0;
                cursor:ew-resize;margin:0">
</div>
<script>
  const rng = document.getElementById('rng'),
        top_ = document.getElementById('topimg'),
        line = document.getElementById('line'),
        knob = document.getElementById('knob');
  rng.addEventListener('input', e => {{
    const v = e.target.value;
    top_.style.clipPath = `inset(0 ${{100 - v}}% 0 0)`;
    line.style.left = v + '%';
    knob.style.left = v + '%';
  }});
</script>
""", height=height)


result = st.session_state.get("result")
if result:
    st.markdown("---")
    st.markdown("## 🎉 Your Artistic Masterpiece")
    st.caption(f"⚙️ {result['summary']} · finished at {result['when']} "
               f"in {result['elapsed']:.1f}s")

    tab_cmp, tab_side, tab_styles = st.tabs(
        ["🔀 Compare (drag the slider)", "◫ Side by side", "🎭 Style references"])

    with tab_cmp:
        comparison_slider(result["content"], result["output"])

    with tab_side:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(result["content"], caption="Original content",
                     use_container_width=True)
        with c2:
            st.image(result["output"], caption="Stylized artwork",
                     use_container_width=True)

    with tab_styles:
        if len(result["styles"]) > 1:
            st.info("💡 Blend weights used (auto weights reflect similarity to your content):")
        cols = st.columns(min(len(result["styles"]), 4))
        for i, (img, w) in enumerate(zip(result["styles"], result["weights"])):
            with cols[i % len(cols)]:
                st.image(img, caption=f"Style {i + 1} · weight {w:.2f}",
                         use_container_width=True)

    st.markdown("### 📥 Download")
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        st.download_button("📥 PNG (lossless)", result["png"],
                           "stylized_output.png", "image/png",
                           use_container_width=True)
    with d2:
        st.download_button("📥 JPEG (optimized)", result["jpg"],
                           "stylized_output.jpg", "image/jpeg",
                           use_container_width=True)
    with d3:
        st.download_button("📥 Before/after comparison", result["cmp"],
                           "style_comparison.png", "image/png",
                           use_container_width=True)

    st.markdown("### 📊 Quality Analysis")
    m = result["metrics"]
    mc = st.columns(4)
    mc[0].metric("🔍 SSIM", f"{m['ssim']:.4f}",
                 help="Structural similarity to the original (0–1, higher = more content kept)")
    mc[1].metric("📡 PSNR", f"{m['psnr']:.2f} dB",
                 help="Peak signal-to-noise ratio vs. the original")
    mc[2].metric("🎨 ΔE (color shift)", f"{m['delta_e']:.2f}",
                 help="Mean perceptual color difference (CIELAB)")
    mc[3].metric("⚡ Style strength", f"{m['style_strength']:.3f}",
                 help="How much visual style was applied (0–1)")

# ------------------------------------------------------------------ history
if st.session_state.history:
    with st.expander(f"🖼️ Session gallery ({len(st.session_state.history)} results)"):
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.history):
            with cols[i % 4]:
                st.image(item["thumb"], caption=item["caption"],
                         use_container_width=True)

# -------------------------------------------------------------------- guide
with st.expander("🎯 Tips & optimal settings"):
    st.markdown("""
| Look you want | α | Content pres. | Passes | Colors | Notes |
|---|---|---|---|---|---|
| **Van Gogh / expressive** | 0.75–0.85 | 0.2–0.3 | 2 | style's | sharpening 1.3–1.5 |
| **Abstract / cubist** | 0.85–0.95 | 0.05–0.2 | 3 | style's | disable guided transfer |
| **Photographic / film** | 0.4–0.6 | 0.4–0.6 | 1 | preserved | keep post-processing neutral |
| **Watercolor / soft** | 0.6–0.75 | 0.3–0.4 | 1–2 | either | sharpening ≤ 1.0 |
| **Portraits** | ≤ 0.6 | ≥ 0.4 | 1 | preserved | guided transfer protects faces |

**Pro tips**
- *Processing resolution* controls detail during stylization; the result is always rendered back at your photo's full resolution. 768px is the quality/speed sweet spot on CPU.
- *Refinement passes* re-stylize the output — the most effective way to get a stronger, more painterly look without artifacts.
- *Preserve original colors* transfers only the style's brushwork/texture (luminance), keeping your photo's palette — great for portraits and product shots.
- With multiple styles, **Auto** weighting favors styles statistically closer to your content; switch to **Manual** for full control.

**Troubleshooting**
- Too stylized / unrecognizable → lower α, raise content preservation.
- Not enough style → raise α or add a refinement pass.
- Oversaturated → lower saturation boost or enable color preservation.
- Slow → drop processing resolution to 384–512px.
""")

# ------------------------------------------------------------------- footer
st.markdown("""
<div class="footer-card">
    <h3 style="color:#667eea;margin:0 0 .6rem">🎨 AI Neural Style Transfer Studio</h3>
    <p style="margin:0;color:#888">Built with ❤️ by Hadeed Ahmad · Streamlit + PyTorch</p>
    <p style="margin:.4rem 0 0;font-size:.9em;color:#999">
        Powered by Adaptive Instance Normalization (AdaIN)</p>
</div>
""", unsafe_allow_html=True)
