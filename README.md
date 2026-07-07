# 🎨 AI Neural Style Transfer Studio

A powerful and user-friendly web application that transforms your photos into artistic masterpieces using advanced neural networks and the AdaIN (Adaptive Instance Normalization) technique.

## ✨ Features

- **🎯 Smart Presets**: One-click configurations that prefill all controls — tweak freely afterwards
- **🔀 Interactive Before/After Slider**: Drag to compare the original and stylized image
- **🎭 Multi-Style Blending**: Mix several styles with automatic (content-aware) or manual weights
- **🔁 Style Refinement Passes**: Re-stylize the output 1–3 times for a stronger, more painterly look
- **🌈 Color Preservation**: Apply the style's texture while keeping your photo's original colors
- **🧠 Guided Style Transfer**: Content-aware protection of edges, faces and text
- **📊 Quality Analysis**: SSIM, PSNR, ΔE and style-strength metrics for every result
- **🎨 Advanced Post-Processing**: Denoise, unsharp mask, CLAHE contrast, saturation (OpenCV with PIL fallback)
- **🖼️ Session Gallery**: Your recent results stay available; downloads never wipe the current result
- **📥 Multiple Download Formats**: PNG, JPEG, and side-by-side comparison images

## 🚀 Quick Start

### Option 1: Streamlit Cloud Deployment (Recommended)

This app is optimized for Streamlit Cloud deployment with the following enhancements:

**Cloud Compatibility Features:**
- ✅ Uses `opencv-python-headless` for cloud environments
- ✅ Includes `packages.txt` for system dependencies
- ✅ Graceful fallbacks when OpenCV is unavailable
- ✅ PIL-based post-processing backup
- ✅ Optimized memory usage for cloud resources

**Deploy to Streamlit Cloud:**
1. Fork this repository to your GitHub account
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select this repository
5. Set main file path: `app.py`
6. Deploy and share your app!

### Option 2: Local Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Or double-click `run_app.bat` on Windows

3. **Open in Browser**: The app will automatically open at `http://localhost:8501`

## 📁 Required Files

Make sure you have these model files in the `models/` folder:
- `decoder.pth` - The decoder neural network
- `vgg_normalised.pth` - The normalized VGG encoder

## 🎨 How to Use

1. **Upload Images**: Choose a content image and one or more style images
2. **Select Preset**: Pick from artistic presets or customize manually
3. **Adjust Parameters**: Fine-tune style strength, content preservation, and post-processing
4. **Generate**: Click the "Create Artistic Masterpiece" button
5. **Download**: Save your results in your preferred format

## 🎯 Optimal Settings Guide

### Van Gogh Style
- Style Strength: 0.75-0.85
- Content Preservation: 0.2-0.3
- Refinement Passes: 2
- Guided Transfer: ✅ Enabled

### Abstract/Picasso Style
- Style Strength: 0.85-0.95
- Content Preservation: 0.05-0.2
- Refinement Passes: 3
- Guided Transfer: ❌ Disabled

### Photographic Styles
- Style Strength: 0.4-0.6
- Content Preservation: 0.4-0.6
- Refinement Passes: 1
- Guided Transfer: ✅ Enabled
- Preserve Original Colors: ✅ Enabled

## 🔧 Technical Details

- **Backend**: PyTorch neural networks (`pipeline.py` — UI-independent, testable core)
- **Frontend**: Streamlit web framework (`app.py` — presentation and state only)
- **Algorithm**: AdaIN (Adaptive Instance Normalization); multiple styles are blended by
  interpolating their AdaIN statistics, so style images of any resolution mix safely
- **Processing**: GPU acceleration when available; dimensions are snapped to the network
  stride so the decoder output always matches the input exactly
- **Image Formats**: JPG, JPEG, PNG, WEBP support (EXIF orientation handled)

## 💡 Pro Tips

1. **For portraits**: Keep content preservation > 0.4 and enable color preservation
2. **For stronger style**: Add refinement passes instead of pushing α to 1.0
3. **For dramatic effects**: Disable guided transfer
4. **Processing size**: 768px offers the best quality/speed balance
5. **Multiple styles**: Auto weighting blends them by content similarity, or set manual weights

## 🚀 Performance

- **GPU Acceleration**: Automatically uses CUDA when available
- **Optimized Processing**: Efficient memory usage and caching
- **Batch Processing**: Support for multiple style images
- **Progressive Updates**: Real-time progress feedback

## 📊 Quality Metrics

The app provides several quality analysis metrics:
- **SSIM Score**: Structural similarity to original
- **PSNR**: Peak signal-to-noise ratio
- **Color Difference (ΔE)**: Perceptual color difference
- **Style Transfer Strength**: How much style was applied

## 🛠️ Troubleshooting

### Common Issues:
- **Blurry output**: Increase sharpening or processing resolution
- **Too stylized**: Decrease style strength, increase content preservation
- **Artifacts**: Reduce sharpening strength or enable noise reduction
- **Slow processing**: Reduce processing resolution or disable post-processing

### Cloud Deployment Issues:
- **OpenCV errors**: The app automatically falls back to PIL-based processing
- **Missing system dependencies**: Ensure `packages.txt` is included in your repo
- **Memory issues**: Use lower processing resolutions (512px) for cloud deployment
- **Import errors**: Check that `requirements.txt` uses `opencv-python-headless`

## 📝 License

This project uses the AdaIN neural style transfer technique. Make sure you have the proper model weights and follow any associated licensing terms.

---

*Built with ❤️ using Streamlit, PyTorch & Advanced AI*
