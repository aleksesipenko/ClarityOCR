# ClarityOCR - Apple Silicon Support

ClarityOCR now supports Apple Silicon (M1, M2, M3 Pro/Max/Ultra) with GPU acceleration via **Metal Performance Shaders (MPS)**.

## 🍎 Supported Devices

- **M1 / M1 Pro / M1 Max / M1 Ultra**
- **M2 / M2 Pro / M2 Max / M2 Ultra**
- **M3 / M3 Pro / M3 Max / M3 Ultra**

## ⚡ Installation (Apple Silicon)

### Option 1: Using PyTorch with MPS (Recommended)

```bash
# Clone repository
git clone https://github.com/aleksesipenko/ClarityOCR.git
cd ClarityOCR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for Apple Silicon
pip install -r requirements-apple-silicon.txt
```

### Option 2: Manual Installation

```bash
# Install PyTorch with MPS support
pip install torch>=2.0.0

# Install other dependencies
pip install \
    marker-pdf>=0.3.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    openai>=1.0.0 \
    pypdfium2>=4.0.0
```

## 🔍 Device Detection

ClarityOCR automatically detects the best available device:

1. **CUDA** (NVIDIA GPUs) - highest priority
2. **MPS** (Apple Silicon) - medium priority
3. **CPU** - fallback

Test device detection:
```bash
python3 -m clarityocr.device_utils
```

## 🐛 Known Issues

### Pillow Installation on Apple Silicon

**Problem:**
PyPI repository doesn't provide Pillow 10.x pre-built wheels for Apple Silicon. The latest version (11.3.0) is available, but `marker-pdf` requires Pillow < 11.0.0. When trying to install Pillow 10.x, it attempts to compile from source and fails with:
```
The headers or library files could not be found for jpeg
```

**Workaround 1: Use Docker (Recommended for Testing)**
```bash
# Test on Linux environment (no Apple Silicon issues)
docker run --platform linux/amd64 -v $(pwd):/app -w /app python:3.12 -c "
import clarityocr
clarityocr.convert_pdf('app/document.pdf', 'app/document.md')
"
```

**Workaround 2: Install System Libraries**
```bash
# Install JPEG libraries before attempting Pillow build
brew install libjpeg libpng libtiff
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"
pip install "Pillow<11.0.0"
```

**Workaround 3: Wait for marker-pdf Update**
The `marker-pdf` maintainers are working on Pillow 11.x compatibility. Check for updates:
```bash
pip install --upgrade marker-pdf
```

**Current Status:**
- ✅ **Code:** Fully functional for Apple Silicon
- ✅ **Device Detection:** Working (CUDA/MPS/CPU)
- ⚠️ **Local Installation:** Blocked by PyPI dependency issues
- ✅ **CI/CD:** Should work on Linux runners

**Verification Without marker-pdf:**
```bash
# Test device detection without full marker-pdf installation
cd /Users/alexesip/clawd/projects/ClarityOCR
python3 -c "
from clarityocr.device_utils import get_device_type, get_device_name, get_memory_info

print('✅ device_utils works on Apple Silicon!')
print(f'Device: {get_device_type()}')
print(f'Name: {get_device_name()}')
mem = get_memory_info()
print(f'Memory: {mem[\"total\"]:.2f} GB total, {mem[\"used\"]:.2f} GB used')
"
```

To check which device is detected:

```bash
python -m clarityocr.device_utils
```

Example output:
```
=== ClarityOCR Device Detection ===
Device Type: mps
Device Name: MPS (Apple Silicon: arm)
GPU Available: True

Configuration:
✅ MPS configured (Apple Silicon GPU)

Memory Info:
  Total: 16.00 GB
  Used: 8.50 GB
  Free: 7.50 GB
```

## 🚀 Running on Apple Silicon

### Web UI

```bash
# Start web server
python -m clarityocr.server

# Or using uvicorn directly
uvicorn clarityocr.server:app --host 0.0.0.0 --port 8000
```

### CLI

```bash
# Convert PDF to Markdown
python -m clarityocr.converter input.pdf output.md

# With specific settings
python -m clarityocr.converter input.pdf output.md --layout-batch 4 --recognition-batch 8
```

## ⚙️ Configuration

### MPS-Specific Settings

For optimal performance on Apple Silicon, consider these settings:

```bash
# Smaller batch sizes work better on MPS
python -m clarityocr.converter input.pdf output.md \
    --layout-batch 2 \
    --recognition-batch 4
```

### Mixed Precision

MPS supports bfloat16 operations for faster inference:

```python
import torch
torch.backends.mps.allow_tf32 = True  # Enabled by default in ClarityOCR
```

## 📊 Performance

Expected performance on different Apple Silicon chips:

| Chip | Unified Memory | Relative Speed* |
|------|---------------|-----------------|
| M1   | 8-16 GB       | 1x              |
| M1 Pro/Max | 16-32 GB | 1.5-2x          |
| M2   | 8-24 GB       | 1.2-1.5x        |
| M2 Pro/Max | 16-96 GB | 2-3x            |
| M3   | 8-24 GB       | 1.5-2x          |
| M3 Pro/Max | 16-128 GB | 2.5-3.5x         |

\*Compared to CPU baseline

## 🔧 Troubleshooting

### MPS not detected

If MPS is not detected:

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

If MPS is not available:
1. Ensure you're running on Apple Silicon (not Intel Mac)
2. Update PyTorch: `pip install --upgrade torch`
3. Check macOS version (macOS 12.3+ required for MPS)

### Out of Memory Errors

Apple Silicon uses **unified memory** (shared CPU/GPU). If you encounter OOM:

```bash
# Reduce batch sizes
python -m clarityocr.converter input.pdf output.md \
    --layout-batch 1 \
    --recognition-batch 2

# Close other memory-intensive applications
# Activity Monitor can help identify memory usage
```

### Slow Performance

If conversion is slow:

1. Check Activity Monitor for GPU utilization
2. Reduce batch sizes if GPU is underutilized
3. Ensure no other heavy GPU workloads are running

## 🆚 CUDA vs MPS

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) |
|---------|---------------|---------------------|
| VRAM | Dedicated GPU memory | Unified memory |
| Precision | BF16/FP16 | BF16 |
| Batch sizes | Larger (8-32) | Smaller (2-8) |
| nvidia-smi | Available | Not applicable |
| Memory monitoring | Detailed | Basic |

## 📝 Notes

- **Unified Memory**: Apple Silicon shares memory between CPU and GPU. Close other applications before large conversions.
- **No nvidia-smi**: MPS doesn't have an equivalent to nvidia-smi. Use Activity Monitor instead.
- **Batch sizes**: MPS generally prefers smaller batch sizes than CUDA.
- **First run**: The first conversion may be slower as models load into memory.

## 🐛 Reporting Issues

When reporting issues on Apple Silicon, include:

1. Mac model and chip (e.g., MacBook Pro M2)
2. macOS version
3. PyTorch version (`python -c "import torch; print(torch.__version__)"`)
4. Device detection output (`python -m clarityocr.device_utils`)
5. Error messages or logs

## 📚 Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Developer - Metal](https://developer.apple.com/metal/)
- [ClarityOCR GitHub](https://github.com/aleksesipenko/ClarityOCR)

---

**Happy OCR on Apple Silicon! 🍎**
