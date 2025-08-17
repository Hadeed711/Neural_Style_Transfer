# 🎨 AI Neural Style Transfer Studio

A powerful and user-friendly web application that transforms your photos into artistic masterpieces using advanced neural networks and the AdaIN (Adaptive Instance Normalization) technique.

## ✨ Features

- **🎯 Smart Presets**: Quick configurations for different artistic styles
- **🔄 Multi-Scale Processing**: Better texture transfer at multiple scales
- **🧠 Guided Style Transfer**: Content-aware style application
- **📊 Quality Analysis**: Technical metrics to evaluate your results
- **🎨 Advanced Post-Processing**: Sharpening, contrast, and saturation controls
- **📥 Multiple Download Formats**: PNG, JPEG, and comparison images
- **💡 Expert Tips**: Built-in guidance for optimal results

## 🚀 Quick Start

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
- Multi-Scale: ✅ Enabled
- Guided Transfer: ✅ Enabled

### Abstract/Picasso Style
- Style Strength: 0.85-0.95
- Content Preservation: 0.1-0.2
- Multi-Scale: ✅ Enabled
- Guided Transfer: ❌ Disabled

### Photographic Styles
- Style Strength: 0.4-0.6
- Content Preservation: 0.4-0.6
- Multi-Scale: ❌ Disabled
- Guided Transfer: ✅ Enabled

## 🔧 Technical Details

- **Backend**: PyTorch neural networks
- **Frontend**: Streamlit web framework
- **Algorithm**: AdaIN (Adaptive Instance Normalization)
- **Processing**: GPU acceleration when available
- **Image Formats**: JPG, JPEG, PNG support

## 💡 Pro Tips

1. **For portraits**: Keep content preservation > 0.4
2. **For landscapes**: Enable multi-scale processing
3. **For dramatic effects**: Disable guided transfer
4. **Processing size**: 768px offers the best quality/speed balance
5. **Multiple styles**: The AI automatically weights and blends them

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

- **Blurry output**: Increase sharpening or processing resolution
- **Too stylized**: Decrease style strength, increase content preservation
- **Artifacts**: Reduce sharpening strength or enable noise reduction
- **Slow processing**: Reduce processing resolution or disable post-processing

## 📝 License

This project uses the AdaIN neural style transfer technique. Make sure you have the proper model weights and follow any associated licensing terms.

---

*Built with ❤️ using Streamlit, PyTorch & Advanced AI*
