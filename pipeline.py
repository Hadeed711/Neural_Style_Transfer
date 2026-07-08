"""Core AdaIN style-transfer pipeline.

UI-independent: everything here works with plain PIL images, numpy arrays and
torch tensors so it can be tested (and reused) without Streamlit.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

import net
from function import calc_mean_std

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Style statistics are global per-channel moments, so a modest style
# resolution loses nothing while keeping encoding fast.
STYLE_SIZE = 512


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(model_dir: str, device: torch.device = None):
    """Load the VGG encoder (up to relu4_1) and the AdaIN decoder.

    Returns (encoder, decoder, device). Raises FileNotFoundError if the
    weight files are missing.
    """
    device = device or get_device()

    vgg_path = os.path.join(model_dir, "vgg_normalised.pth")
    dec_path = os.path.join(model_dir, "decoder.pth")
    for path in (vgg_path, dec_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    vgg = net.vgg
    vgg.load_state_dict(torch.load(vgg_path, map_location="cpu", weights_only=True))
    encoder = nn.Sequential(*list(vgg.children())[:31]).to(device).eval()

    decoder = net.decoder
    decoder.load_state_dict(torch.load(dec_path, map_location="cpu", weights_only=True))
    decoder = decoder.to(device).eval()

    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    return encoder, decoder, device


def _to_tensor(img: Image.Image, max_side: int, device: torch.device) -> torch.Tensor:
    """Resize so the longest side is at most max_side, snap both dimensions to
    multiples of 8 (the encoder/decoder stride) and convert to a 1xCxHxW tensor.

    Snapping guarantees the decoder output has exactly the input resolution,
    so no cropping/padding artifacts appear at the borders.
    """
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(8, int(round(w * scale / 8)) * 8)
    new_h = max(8, int(round(h * scale / 8)) * 8)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return TF.to_tensor(img).unsqueeze(0).to(device)


def _resolve_weights(c_feat: torch.Tensor, stats, style_weights):
    """Normalize user weights, or derive content-aware weights from the cosine
    similarity between content and style channel statistics."""
    n = len(stats)
    if n == 1:
        return [1.0]
    if style_weights is not None:
        total = float(sum(style_weights))
        if total <= 0:
            return [1.0 / n] * n
        return [float(w) / total for w in style_weights]

    c_vec = c_feat.mean(dim=[2, 3])
    raw = []
    for mean, _ in stats:
        s_vec = mean.squeeze(-1).squeeze(-1)
        sim = torch.cosine_similarity(c_vec, s_vec, dim=1).mean().item()
        raw.append(max(0.1, abs(sim)))
    total = sum(raw)
    return [w / total for w in raw]


def _guided_blend(content_feat: torch.Tensor, stylized_feat: torch.Tensor,
                  strength: float) -> torch.Tensor:
    """Blend content features back in where they carry strong structure
    (edges, faces, text), keeping full stylization in flat regions.

    The activation magnitude map is standardized before the sigmoid so the
    gate actually varies across the image instead of saturating.
    """
    strength = float(min(max(strength, 0.0), 1.0))
    if strength == 0.0:
        return stylized_feat
    mag = torch.norm(content_feat, dim=1, keepdim=True)
    norm = (mag - mag.mean()) / (mag.std() + 1e-5)
    gate = torch.sigmoid(2.0 * norm) * strength
    return gate * content_feat + (1.0 - gate) * stylized_feat


def stylize(encoder, decoder, content_img: Image.Image, style_imgs,
            *, alpha: float = 0.7, processing_size: int = 512, passes: int = 1,
            guided: bool = True, guidance: float = 0.3,
            style_weights=None, preserve_color: bool = False,
            device: torch.device = None):
    """Run AdaIN style transfer.

    Multiple styles are blended by interpolating their AdaIN statistics with
    the given (or content-aware) weights — equivalent to interpolating the
    stylized features, but independent of the style images' resolutions.
    Extra passes re-encode the result and stylize again for a stronger effect.

    Returns (output PIL image at content_img's original size, weights used).
    """
    device = device or next(decoder.parameters()).device
    alpha = float(min(max(alpha, 0.0), 1.0))
    c_ten = _to_tensor(content_img, processing_size, device)

    with torch.inference_mode():
        stats = []
        for s_img in style_imgs:
            s_feat = encoder(_to_tensor(s_img, STYLE_SIZE, device))
            stats.append(calc_mean_std(s_feat))
            del s_feat

        c_feat = encoder(c_ten)
        weights = _resolve_weights(c_feat, stats, style_weights)
        s_mean = sum(w * m for w, (m, _) in zip(weights, stats))
        s_std = sum(w * s for w, (_, s) in zip(weights, stats))

        out = c_ten
        for i in range(max(1, int(passes))):
            feat = c_feat if i == 0 else encoder(out)
            c_mean, c_std = calc_mean_std(feat)
            t = (feat - c_mean) / c_std * s_std + s_mean
            if guided:
                t = _guided_blend(feat, t, guidance)
            t = alpha * t + (1.0 - alpha) * feat
            out = decoder(t).clamp_(0.0, 1.0)

    out_img = TF.to_pil_image(out.squeeze(0).cpu())
    if out_img.size != content_img.size:
        out_img = out_img.resize(content_img.size, Image.Resampling.LANCZOS)
    if preserve_color:
        out_img = transfer_color(content_img, out_img)
    return out_img, weights


def transfer_color(content_img: Image.Image, stylized_img: Image.Image) -> Image.Image:
    """Keep the content image's colors: take luminance from the stylized
    result and chrominance from the original (YCbCr)."""
    if content_img.size != stylized_img.size:
        content_img = content_img.resize(stylized_img.size, Image.Resampling.LANCZOS)
    y = stylized_img.convert("YCbCr").split()[0]
    _, cb, cr = content_img.convert("YCbCr").split()
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def post_process(img_array: np.ndarray, *, sharpen: float = 1.2,
                 contrast: float = 1.1, saturation: float = 1.0,
                 noise_reduction: bool = True) -> np.ndarray:
    """Post-processing on a uint8 RGB array. Uses OpenCV (bilateral denoise,
    unsharp mask, CLAHE) when available, otherwise falls back to PIL."""
    if CV2_AVAILABLE:
        try:
            return _cv2_post_process(img_array, sharpen, contrast, saturation,
                                     noise_reduction)
        except Exception:
            pass
    return _pil_post_process(img_array, sharpen, contrast, saturation)


def _cv2_post_process(img_array, sharpen, contrast, saturation, noise_reduction):
    img_f = img_array.astype(np.float32) / 255.0

    if noise_reduction:
        # Near-zero sigmaSpace keeps this edge-preserving without visibly
        # softening the image (matches the original tuning).
        img_f = cv2.bilateralFilter(img_f, d=9, sigmaColor=0.08, sigmaSpace=0.08)

    if sharpen != 1.0:
        blur = cv2.GaussianBlur(img_f, (0, 0), 1.5)
        if sharpen > 1.0:  # unsharp mask
            amt = (sharpen - 1.0) * 0.8
            img_f = cv2.addWeighted(img_f, 1.0 + amt, blur, -amt, 0)
        else:  # soften
            img_f = cv2.addWeighted(img_f, sharpen, blur, 1.0 - sharpen, 0)
        img_f = np.clip(img_f, 0.0, 1.0)

    if contrast > 1.0:  # local contrast via CLAHE on the L channel
        u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0 * contrast, tileGridSize=(8, 8))
        lab = cv2.merge([clahe.apply(l), a, b])
        img_f = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    elif contrast < 1.0:  # linear contrast reduction
        mean = img_f.mean()
        img_f = np.clip(mean + (img_f - mean) * contrast, 0.0, 1.0)

    if saturation != 1.0:
        u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img_f = cv2.cvtColor(hsv.astype(np.uint8),
                             cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def _pil_post_process(img_array, sharpen, contrast, saturation):
    img = Image.fromarray(img_array)
    if sharpen != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpen)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    return np.asarray(img)


def compute_metrics(content_img: Image.Image, output_img: Image.Image) -> dict:
    """SSIM, PSNR, mean CIELAB ΔE and a derived style-strength score,
    computed on copies downscaled to at most 512px for speed."""
    from skimage.metrics import structural_similarity as ssim

    size = (min(512, content_img.width), min(512, content_img.height))
    a = np.asarray(content_img.resize(size, Image.Resampling.LANCZOS),
                   dtype=np.float32) / 255.0
    b = np.asarray(output_img.resize(size, Image.Resampling.LANCZOS),
                   dtype=np.float32) / 255.0

    win = min(7, min(size))
    if win % 2 == 0:
        win -= 1
    try:
        ssim_score = float(ssim(a, b, channel_axis=2, data_range=1.0,
                                win_size=max(3, win)))
    except Exception:
        ssim_score = 0.0

    mse = float(np.mean((a - b) ** 2))
    psnr = 100.0 if mse <= 1e-12 else float(-10 * np.log10(mse))

    if CV2_AVAILABLE:
        lab_a = cv2.cvtColor((a * 255).astype(np.uint8),
                             cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor((b * 255).astype(np.uint8),
                             cv2.COLOR_RGB2LAB).astype(np.float32)
        delta_e = float(np.mean(np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=2))))
    else:
        delta_e = float(np.mean(np.sqrt(np.sum((a - b) ** 2, axis=2))) * 100)

    return {
        "ssim": ssim_score,
        "psnr": psnr,
        "delta_e": delta_e,
        "style_strength": min(1.0, delta_e / 50.0),
    }
