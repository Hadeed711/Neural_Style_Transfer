import os, traceback, streamlit as st
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import gaussian_blur
from PIL import Image, ImageFilter, ImageEnhance

# Try to import OpenCV with fallback for cloud deployment
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Some advanced post-processing features will be disabled.")

# Set page configuration for better appearance
st.set_page_config(
    page_title="AI Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
}
.stProgress .st-bo {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}
.upload-box {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    background: rgba(102, 126, 234, 0.1);
}
</style>
""", unsafe_allow_html=True)

cwd = os.getcwd()

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🎨 AI Neural Style Transfer Studio</h1>
    <p style="font-size: 1.2em; margin: 0;">Transform your images with the magic of artificial intelligence and artistic style</p>
</div>
""", unsafe_allow_html=True)

# — Upload widgets —
st.markdown("## 📤 Upload Your Images")
st.markdown("*Choose a content image and one or more style images to create your masterpiece*")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 📷 Content Image")
    content_file = st.file_uploader(
        "Upload your main image", 
        type=['jpg','jpeg','png'],
        help="This is the image that will be transformed with the artistic style"
    )
    
with col2:
    st.markdown("### 🎭 Style Images")
    style_files = st.file_uploader(
        "Upload artistic style references", 
        type=['jpg','jpeg','png'], 
        accept_multiple_files=True,
        help="Upload one or more artistic images to apply their style to your content"
    )

# Initialize models silently
@st.cache_resource
def load_models():
    """Load neural network models with caching for performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load VGG encoder
        from net import vgg as adain_vgg
        adain_vgg = adain_vgg.to(device)
        vgg_path = os.path.join(cwd, "models", "vgg_normalised.pth")
        adain_vgg.load_state_dict(torch.load(vgg_path, map_location=device))
        adain_vgg.eval()
        
        # Build encoder up to relu4_1
        encoder = nn.Sequential(*list(adain_vgg.children())[:31]).to(device).eval()
        
        # Load decoder
        from net import decoder as official_decoder
        decoder = official_decoder.to(device).eval()
        decoder_path = os.path.join(cwd, "models", "decoder.pth")
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        return encoder, decoder, device
        
    except Exception as e:
        st.error(f"❌ Failed to load neural network models: {str(e)}")
        st.error("Please ensure both `vgg_normalised.pth` and `decoder.pth` are in the `models/` folder")
        st.stop()

encoder, decoder, device = load_models()

# Show device info
device_icon = "🚀" if device.type == "cuda" else "💻"
st.sidebar.success(f"{device_icon} Running on: {device.type.upper()}")

# — Enhanced AdaIN function with better feature matching —
def adaptive_instance_normalization(c_feat, s_feat, eps=1e-5):
    """Enhanced AdaIN with better statistical matching"""
    # Calculate statistics
    c_mean = torch.mean(c_feat, [2,3], keepdim=True)
    c_std  = torch.std(c_feat, [2,3], keepdim=True, unbiased=False) + eps
    s_mean = torch.mean(s_feat, [2,3], keepdim=True)
    s_std  = torch.std(s_feat, [2,3], keepdim=True, unbiased=False) + eps
    
    # Normalize content features
    normalized = (c_feat - c_mean) / c_std
    
    # Apply style statistics with improved blending
    stylized = normalized * s_std + s_mean
    
    return stylized

def multi_scale_adain(c_feat, s_feat, scales=[1.0, 0.7, 0.4]):
    """Multi-scale AdaIN for better texture transfer"""
    results = []
    
    for scale in scales:
        if scale != 1.0:
            # Downsample features
            h, w = c_feat.shape[2], c_feat.shape[3]
            new_h, new_w = int(h * scale), int(w * scale)
            c_scaled = torch.nn.functional.interpolate(c_feat, size=(new_h, new_w), mode='bilinear', align_corners=False)
            s_scaled = torch.nn.functional.interpolate(s_feat, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Apply AdaIN
            stylized_scaled = adaptive_instance_normalization(c_scaled, s_scaled)
            
            # Upsample back
            stylized_upscaled = torch.nn.functional.interpolate(stylized_scaled, size=(h, w), mode='bilinear', align_corners=False)
            results.append(stylized_upscaled)
        else:
            results.append(adaptive_instance_normalization(c_feat, s_feat))
    
    # Improved weighted combination
    weights = [0.6, 0.25, 0.15]  # Favor original scale but include details
    combined = sum(w * r for w, r in zip(weights, results))
    return combined

def guided_style_transfer(c_feat, s_feat, guidance_strength=0.3):
    """Guided style transfer with content-aware adaptation"""
    # Calculate content importance map
    content_magnitude = torch.norm(c_feat, dim=1, keepdim=True)
    content_weight = torch.sigmoid(5 * (content_magnitude - content_magnitude.mean()))
    
    # Standard AdaIN
    standard_adain = adaptive_instance_normalization(c_feat, s_feat)
    
    # Content-guided blending
    guided_result = content_weight * c_feat * guidance_strength + (1 - content_weight * guidance_strength) * standard_adain
    
    return guided_result

# — Advanced Stylization Parameters —
st.sidebar.markdown("## 🎨 Style Parameters")
st.sidebar.markdown("*Customize your artistic transformation*")
def load_image(img: Image.Image, resize=False, target_size=512):
    """Enhanced image loading with better preprocessing"""
    if resize:
        # Maintain aspect ratio while resizing
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
    transform_steps = [
        transforms.ToTensor(),  # Convert to [0,1]
    ]
    
    tensor = transforms.Compose(transform_steps)(img).unsqueeze(0).to(device)
    return tensor

def tensor_to_np(tensor):
    """Enhanced tensor to numpy conversion with better handling"""
    arr = tensor.cpu().clamp(0, 1).squeeze(0).numpy().transpose(1,2,0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (arr * 255).astype(np.uint8)

def advanced_post_processing(img_array, sharpen=1.2, contrast=1.1, saturation=1.0, noise_reduction=True):
    """Advanced post-processing pipeline with fallback for cloud deployment"""
    if not CV2_AVAILABLE:
        # Fallback processing using PIL when OpenCV is not available
        return pil_post_processing(img_array, sharpen, contrast, saturation)
    
    try:
        # Convert to float for processing
        img_float = img_array.astype(np.float32) / 255.0
        
        # Noise reduction with edge preservation
        if noise_reduction:
            img_float = cv2.bilateralFilter(img_float, d=9, sigmaColor=0.08, sigmaSpace=0.08)
        
        # Advanced sharpening with unsharp mask
        if sharpen > 1.0:
            gaussian = cv2.GaussianBlur(img_float, (0, 0), 1.5)
            img_float = cv2.addWeighted(img_float, 1.0 + (sharpen - 1.0) * 0.8, gaussian, -(sharpen - 1.0) * 0.8, 0)
        
        # CLAHE for local contrast enhancement
        if contrast != 1.0:
            img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0 * contrast, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l, a, b])
            img_float = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Saturation adjustment in HSV space
        if saturation != 1.0:
            img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,1] = hsv[:,:,1] * saturation
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            img_float = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        return np.clip(img_float * 255, 0, 255).astype(np.uint8)
        
    except Exception as e:
        st.warning(f"Advanced post-processing failed, using fallback: {str(e)}")
        return pil_post_processing(img_array, sharpen, contrast, saturation)

def pil_post_processing(img_array, sharpen=1.2, contrast=1.1, saturation=1.0):
    """Fallback post-processing using PIL when OpenCV is not available"""
    try:
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array)
        
        # Apply sharpening
        if sharpen > 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpen)
        
        # Apply contrast enhancement
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # Apply saturation adjustment
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        
        # Convert back to numpy array
        return np.array(img)
        
    except Exception as e:
        st.warning(f"PIL post-processing failed: {str(e)}")
        return img_array

# — Enhanced Helper Functions —
def load_image(img: Image.Image, resize=False, target_size=512):
    """Enhanced image loading with better preprocessing"""
    if resize:
        # Maintain aspect ratio while resizing
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
    transform_steps = [
        transforms.ToTensor(),  # Convert to [0,1]
    ]
    
    tensor = transforms.Compose(transform_steps)(img).unsqueeze(0).to(device)
    return tensor

def tensor_to_np(tensor):
    """Enhanced tensor to numpy conversion with better handling"""
    arr = tensor.cpu().clamp(0, 1).squeeze(0).numpy().transpose(1,2,0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (arr * 255).astype(np.uint8)

# — Advanced Stylization Parameters —
st.sidebar.markdown("## 🎨 Style Parameters")
st.sidebar.markdown("*Customize your artistic transformation*")

# Preset configurations
preset = st.sidebar.selectbox("🎯 Quick Presets", [
    "Custom", "Artistic Strong", "Artistic Subtle", "Photographic", 
    "Abstract", "Texture Heavy", "Content Preserve"
])

# Preset values
preset_configs = {
    "Artistic Strong": {"alpha": 0.85, "content_weight": 0.15, "multi_scale": True, "guided": True, "sharpen": 1.4, "contrast": 1.2, "saturation": 1.1},
    "Artistic Subtle": {"alpha": 0.6, "content_weight": 0.4, "multi_scale": True, "guided": True, "sharpen": 1.1, "contrast": 1.05, "saturation": 1.05},
    "Photographic": {"alpha": 0.5, "content_weight": 0.5, "multi_scale": False, "guided": True, "sharpen": 1.0, "contrast": 1.0, "saturation": 0.95},
    "Abstract": {"alpha": 0.9, "content_weight": 0.1, "multi_scale": True, "guided": False, "sharpen": 1.6, "contrast": 1.3, "saturation": 1.2},
    "Texture Heavy": {"alpha": 0.8, "content_weight": 0.2, "multi_scale": True, "guided": True, "sharpen": 1.5, "contrast": 1.15, "saturation": 1.0},
    "Content Preserve": {"alpha": 0.4, "content_weight": 0.7, "multi_scale": False, "guided": True, "sharpen": 0.9, "contrast": 1.0, "saturation": 1.0}
}

if preset != "Custom":
    config = preset_configs[preset]
    alpha = config["alpha"]
    content_weight = config["content_weight"]
    multi_scale = config["multi_scale"]
    guided_transfer = config["guided"]
    sharpen_strength = config["sharpen"]
    contrast_boost = config["contrast"]
    saturation_boost = config["saturation"]
    
    st.sidebar.info(f"Using {preset} preset")
else:
    alpha = st.sidebar.slider("Style Strength (α)", 0.0, 1.0, 0.7, 0.05, 
                             help="Higher values = more style, less content")
    content_weight = st.sidebar.slider("Content Preservation", 0.0, 1.0, 0.3, 0.05,
                                     help="How much original content to preserve")
    multi_scale = st.sidebar.checkbox("Multi-Scale Processing", value=True,
                                    help="Better texture transfer at multiple scales")
    guided_transfer = st.sidebar.checkbox("Guided Style Transfer", value=True,
                                        help="Content-aware style application")

preserve_size = st.sidebar.checkbox("Preserve Original Image Size", value=True)
enhance_quality = st.sidebar.checkbox("Advanced Quality Enhancement", value=True)

st.sidebar.header("🔧 Post-Processing")
if preset == "Custom":
    sharpen_strength = st.sidebar.slider("Sharpening Strength", 0.0, 3.0, 1.2, 0.1)
    contrast_boost = st.sidebar.slider("Contrast Enhancement", 0.5, 2.0, 1.1, 0.05)
    saturation_boost = st.sidebar.slider("Saturation Boost", 0.5, 2.0, 1.0, 0.05)

noise_reduction = st.sidebar.checkbox("Noise Reduction", value=True)

# Processing size selection
processing_size = st.sidebar.selectbox("🔧 Processing Resolution", 
                                     [512, 768, 1024], 
                                     index=1,
                                     help="Higher = better quality but slower")

# Generate button with enhanced styling
st.markdown("---")
generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
with generate_col2:
    generate_button = st.button(
        "🚀 Create Artistic Masterpiece", 
        type="primary", 
        use_container_width=True,
        help="Click to start the neural style transfer process"
    )
if generate_button:
    if not content_file or not style_files:
        st.error("🚫 Please upload both a content image and at least one style image to proceed.")
        st.info("💡 **Tip:** Choose high-quality images for the best results!")
    else:
        # Enhanced processing with better UX
        with st.spinner("🎨 Creating your artistic masterpiece..."):
            status_container = st.container()
            progress_bar = st.progress(0)
            
            with status_container:
                status_text = st.empty()
                
            status_text.text("📥 Loading and preprocessing images...")
            progress_bar.progress(10)
            # Load images with enhanced preprocessing
            cont_img = Image.open(content_file).convert("RGB")
            original_size = cont_img.size
            
            status_text.text("🎭 Preparing style references...")
            # Prepare style images
            style_imgs = [Image.open(f).convert("RGB") for f in style_files]
            progress_bar.progress(20)
            
            status_text.text("🔄 Converting to neural network format...")
            # Load tensors with better sizing
            target_size = processing_size if not preserve_size else min(processing_size, max(original_size))
            cont_ten = load_image(cont_img, resize=not preserve_size, target_size=target_size)
            style_tens = [load_image(img, resize=not preserve_size, target_size=target_size) for img in style_imgs]
            progress_bar.progress(30)

            # Enhanced stylization process
            status_text.text("🧠 Extracting neural features...")
            with torch.no_grad():
                # Extract features
                c_feat = encoder(cont_ten)
                s_feats = [encoder(sty) for sty in style_tens]
                progress_bar.progress(50)
                
                status_text.text("🎨 Applying artistic style transfer...")
                # Multi-style blending with advanced techniques
                if len(s_feats) > 1:
                    # Compute style similarity weights
                    style_weights = []
                    for s_feat in s_feats:
                        # Calculate feature correlation
                        c_flat = c_feat.view(c_feat.size(0), c_feat.size(1), -1)
                        s_flat = s_feat.view(s_feat.size(0), s_feat.size(1), -1)
                        
                        correlation = torch.cosine_similarity(
                            c_flat.mean(dim=2), s_flat.mean(dim=2), dim=1
                        ).mean().item()
                        style_weights.append(max(0.1, abs(correlation)))
                    
                    # Normalize weights
                    total_weight = sum(style_weights)
                    style_weights = [w/total_weight for w in style_weights]
                    
                    # Weighted style feature combination
                    combined_style = sum(w * s for w, s in zip(style_weights, s_feats))
                else:
                    combined_style = s_feats[0]
                    style_weights = [1.0]
                
                progress_bar.progress(60)
                
                status_text.text("✨ Blending content and style...")
                # Apply enhanced AdaIN based on settings
                if guided_transfer:
                    t = guided_style_transfer(c_feat, combined_style, guidance_strength=content_weight)
                elif multi_scale:
                    t = multi_scale_adain(c_feat, combined_style)
                else:
                    t = adaptive_instance_normalization(c_feat, combined_style)
                
                # Final blending
                t = alpha * t + (1 - alpha) * c_feat
                
                progress_bar.progress(70)
                
                status_text.text("🖼️ Generating final artwork...")
                # Generate output
                stylized = decoder(t)
                stylized = torch.clamp(stylized, 0, 1).squeeze(0)
                progress_bar.progress(80)

            # Convert to displayable image
            status_text.text("🔧 Applying final touches...")
            out_np = tensor_to_np(stylized)
            progress_bar.progress(90)
            
            # Advanced post-processing
            if enhance_quality:
                out_np = advanced_post_processing(
                    out_np, 
                    sharpen=sharpen_strength,
                    contrast=contrast_boost,
                    saturation=saturation_boost,
                    noise_reduction=noise_reduction
                )
            
            out_img = Image.fromarray(out_np)
            
            # Resize back to original if we preserved size
            if preserve_size and out_img.size != original_size:
                out_img = out_img.resize(original_size, Image.Resampling.LANCZOS)

            status_text.text("✅ Masterpiece complete!")
            progress_bar.progress(100)

        # Clear the status text after completion
        status_text.empty()
        
        # ==== Enhanced Results Display ====
        st.markdown("---")
        st.markdown("## 🎉 Your Artistic Masterpiece is Ready!")
        
        # Main result display with enhanced layout
        result_col1, result_col2 = st.columns(2, gap="large")
        with result_col1:
            st.markdown("### 📷 Original Content")
            st.image(cont_img, caption="Your original image", use_container_width=True)
        with result_col2:
            st.markdown("### 🎨 Stylized Artwork")
            st.image(out_img, caption="AI-generated artistic transformation", use_container_width=True)

        # Style images preview with weights
        if len(style_imgs) > 1:
            st.markdown("### 🎭 Style References Used")
            st.info("💡 **Smart Blending:** Weights were automatically calculated based on content similarity")
            
            style_cols = st.columns(min(len(style_imgs), 4))
            for i, (img, weight) in enumerate(zip(style_imgs[:4], style_weights[:4])):
                with style_cols[i]:
                    st.image(img, caption=f"Style {i+1} (Weight: {weight:.2f})", use_container_width=True)
        elif len(style_imgs) == 1:
            st.markdown("### 🎭 Style Reference")
            style_col1, style_col2, style_col3 = st.columns([1, 2, 1])
            with style_col2:
                st.image(style_imgs[0], caption="Applied artistic style", use_container_width=True)

        # Download options with enhanced styling
        st.markdown("---")
        st.markdown("### 📥 Download Your Masterpiece")
        st.markdown("*Choose your preferred format and quality*")
        
        download_col1, download_col2, download_col3 = st.columns(3, gap="medium")
        
        with download_col1:
            buf_png = BytesIO()
            out_img.save(buf_png, "PNG", quality=95)
            st.download_button(
                "📥 Download PNG (Highest Quality)", 
                buf_png.getvalue(), 
                "stylized_output.png",
                "image/png",
                use_container_width=True
            )
        
        with download_col2:
            buf_jpg = BytesIO()
            out_img.save(buf_jpg, "JPEG", quality=95, optimize=True)
            st.download_button(
                "📥 Download JPEG (Optimized)", 
                buf_jpg.getvalue(), 
                "stylized_output.jpg",
                "image/jpeg",
                use_container_width=True
            )
            
        with download_col3:
            # Create a comparison image
            max_height = max(cont_img.height, out_img.height)
            comparison = Image.new('RGB', (cont_img.width + out_img.width + 10, max_height), color='white')
            comparison.paste(cont_img, (0, (max_height - cont_img.height) // 2))
            comparison.paste(out_img, (cont_img.width + 10, (max_height - out_img.height) // 2))
            
            buf_comp = BytesIO()
            comparison.save(buf_comp, "PNG", quality=95)
            st.download_button(
                "📥 Download Comparison", 
                buf_comp.getvalue(), 
                "style_comparison.png",
                "image/png",
                use_container_width=True
            )

        # Enhanced quality metrics
        st.markdown("---")
        st.markdown("## 📊 Quality Analysis")
        st.markdown("*Technical analysis of your artistic transformation*")
        
        # Resize for fair comparison
        comp_size = (min(512, original_size[0]), min(512, original_size[1]))
        orig_resized = cont_img.resize(comp_size, Image.Resampling.LANCZOS)
        out_resized = out_img.resize(comp_size, Image.Resampling.LANCZOS)
        
        orig_arr = np.array(orig_resized, dtype=np.float32) / 255.0
        res_arr = np.array(out_resized, dtype=np.float32) / 255.0
        
        # Multiple quality metrics
        try:
            ssim_score = ssim(orig_arr, res_arr, channel_axis=2, data_range=1.0, win_size=min(7, min(orig_arr.shape[:2])//2))
        except:
            ssim_score = 0.0
            
        # PSNR calculation
        mse = np.mean((orig_arr - res_arr) ** 2)
        psnr_score = -10 * np.log10(mse) if mse > 0 else 100
        
        # Color difference metrics
        try:
            if CV2_AVAILABLE:
                orig_uint8 = (orig_arr * 255).astype(np.uint8)
                res_uint8 = (res_arr * 255).astype(np.uint8)
                
                lab_orig = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
                lab_res = cv2.cvtColor(res_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
                delta_e = np.mean(np.sqrt(np.sum((lab_orig - lab_res) ** 2, axis=2)))
            else:
                # Fallback RGB difference calculation
                delta_e = np.mean(np.sqrt(np.sum((orig_arr - res_arr) ** 2, axis=2))) * 100
        except:
            delta_e = np.mean(np.sqrt(np.sum((orig_arr - res_arr) ** 2, axis=2))) * 100
        
        # Display metrics with enhanced presentation
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("🔍 SSIM Score", f"{ssim_score:.4f}", 
                     delta=f"{ssim_score-0.5:.3f}" if ssim_score > 0 else None,
                     help="Structural similarity (higher is better, 0-1)")
        with metric_cols[1]:
            st.metric("📡 PSNR", f"{psnr_score:.2f} dB", 
                     delta=f"{psnr_score-20:.1f}" if psnr_score < 100 else None,
                     help="Peak signal-to-noise ratio")
        with metric_cols[2]:
            st.metric("🎨 Color Difference (ΔE)", f"{delta_e:.2f}", 
                     help="Perceptual color difference (lower is more similar)")
        with metric_cols[3]:
            style_strength = min(1, delta_e / 50)
            st.metric("⚡ Style Transfer Strength", f"{style_strength:.3f}", 
                     help="How much style was applied (0-1)")

# Perfect slider combinations guide
with st.expander("🎯 Optimal Settings Guide - Master the Art of Style Transfer"):
    st.markdown("""
    ## 🎨 Perfected Settings for Different Artistic Styles
    
    ### 🖼️ **Van Gogh Style (Starry Night, Sunflowers)**
    - **Style Strength (α)**: 0.75-0.85
    - **Content Preservation**: 0.2-0.3
    - **Multi-Scale**: ✅ Enabled
    - **Guided Transfer**: ✅ Enabled
    - **Sharpening**: 1.3-1.5
    - **Contrast**: 1.15-1.25
    - **Saturation**: 1.1-1.2
    
    ### 🎭 **Abstract/Picasso Style (Cubism, Modern Art)**
    - **Style Strength (α)**: 0.85-0.95
    - **Content Preservation**: 0.1-0.2
    - **Multi-Scale**: ✅ Enabled
    - **Guided Transfer**: ❌ Disabled
    - **Sharpening**: 1.5-2.0
    - **Contrast**: 1.2-1.4
    - **Saturation**: 1.1-1.3
    
    ### 📸 **Photographic/Realistic Styles (Film, Vintage)**
    - **Style Strength (α)**: 0.4-0.6
    - **Content Preservation**: 0.4-0.6
    - **Multi-Scale**: ❌ Disabled
    - **Guided Transfer**: ✅ Enabled
    - **Sharpening**: 0.9-1.1
    - **Contrast**: 1.0-1.05
    - **Saturation**: 0.95-1.05
    
    ### 🌊 **Watercolor/Soft Styles (Impressionism)**
    - **Style Strength (α)**: 0.6-0.75
    - **Content Preservation**: 0.3-0.4
    - **Multi-Scale**: ✅ Enabled
    - **Guided Transfer**: ✅ Enabled
    - **Sharpening**: 0.8-1.0
    - **Contrast**: 1.0-1.1
    - **Saturation**: 1.05-1.15
    
    ### ⚡ **High Detail/Texture Styles (Gothic, Ornate)**
    - **Style Strength (α)**: 0.8-0.9
    - **Content Preservation**: 0.15-0.25
    - **Multi-Scale**: ✅ Enabled
    - **Guided Transfer**: ✅ Enabled
    - **Sharpening**: 1.4-1.8
    - **Contrast**: 1.15-1.3
    - **Saturation**: 1.0-1.1
    
    ### 💡 **Pro Tips from AI Artists**
    1. **For portraits**: Keep content preservation > 0.4 to maintain facial features
    2. **For landscapes**: Multi-scale processing captures both broad strokes and fine details
    3. **For text/documents**: Use guided transfer + high content preservation (0.6+)
    4. **For dramatic effects**: Disable guided transfer for more abstract results
    5. **Processing size**: 768px offers the sweet spot for quality vs. speed
    6. **Multiple styles**: The AI automatically weights styles - experiment with contrasting art!
    """)

# Information and tips
with st.expander("💡 Advanced Tips & Troubleshooting - Become a Style Transfer Expert"):
    st.markdown("""
    ### 🎯 **For Professional-Quality Results:**
    - **📐 Image Resolution**: Use high-resolution images (1024px+ on longest side) for gallery-quality output
    - **💡 Style Selection**: Choose styles with similar lighting conditions to your content for harmonious results
    - **⚙️ Processing Size**: 768px offers optimal quality/speed balance for most use cases
    - **🎨 Multiple Styles**: The AI intelligently blends multiple styles based on content similarity
    
    ### 🔧 **Parameter Mastery:**
    - **🎚️ Alpha (Style Strength)**: Start with 0.7, increase for more artistic flair, decrease for subtle effects
    - **🛡️ Content Preservation**: Use 0.4+ for faces and important details, 0.2 for abstract transformations
    - **🔄 Multi-Scale**: Essential for capturing fine textures and broad artistic strokes simultaneously
    - **🎯 Guided Transfer**: Enables content-aware style application for more intelligent results
    
    ### 🚀 **Performance Optimization:**
    - Use 512px processing size for quick previews and experiments
    - Disable noise reduction for faster processing on clean images
    - Choose JPEG download for smaller file sizes when sharing online
    
    ### 🔧 **Troubleshooting Common Issues:**
    - **🔍 Artifacts/Noise**: Reduce sharpening strength or enable noise reduction
    - **😵 Too stylized/unrecognizable**: Decrease alpha, increase content preservation
    - **😴 Not enough style impact**: Increase alpha, enable multi-scale processing
    - **🌈 Oversaturated colors**: Reduce saturation boost or choose different style images
    - **😕 Blurry output**: Increase sharpening, check processing resolution, or use higher quality input images
    - **⏱️ Slow processing**: Reduce processing resolution or disable advanced post-processing
    
    ### 🎨 **Creative Techniques:**
    - **Layer styles**: Use multiple complementary style images for unique blends
    - **Seasonal themes**: Match style lighting with content season/time of day
    - **Texture experiments**: Try architectural styles on natural images and vice versa
    - **Portrait enhancement**: Use artistic portrait styles on your photos for professional results
    """)

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; margin-top: 2rem;">
    <h3 style="color: #667eea; margin-bottom: 1rem;">🎨 AI Neural Style Transfer Studio</h3>
    <p style="margin: 0; color: #666;">Built with ❤️ using Streamlit, PyTorch & Advanced AI • Transform your world with artificial creativity</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; color: #888;">Powered by Adaptive Instance Normalization (AdaIN) Neural Networks</p>
</div>
""", unsafe_allow_html=True)