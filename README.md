# ClarityOCR

**High-quality PDF to Markdown conversion with GPU-accelerated OCR and LLM post-processing**

ClarityOCR is a standalone tool for converting PDF documents to clean Markdown text. It combines state-of-the-art OCR (via marker-pdf) with optional LLM-based error correction, all wrapped in an intuitive Web UI.

## Features

- **GPU-Accelerated OCR**: Uses marker-pdf with Surya models for accurate text extraction
- **LLM Post-Processing**: Optional error correction via local LLM (LM Studio compatible)
- **Real-Time Web UI**: Beautiful interface with live progress tracking and GPU stats
- **Batch Processing**: Convert multiple PDFs with automatic queue management
- **Smart Page Markers**: Stable `[p:N]` markers for citation references
- **Encoding Fix**: Automatic mojibake detection and repair (CP1251/Latin-1)

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8+ GB VRAM OR Apple Silicon (M1/M2/M3) for MPS acceleration
- **RAM**: 16+ GB recommended
- **Storage**: ~15 GB for models (downloaded automatically)

### Software
- Python 3.10-3.12 (3.13+ not supported by PyTorch CUDA yet)
- CUDA 12.1+ (for NVIDIA GPU acceleration) OR macOS 12.3+ (for Apple Silicon MPS)
- Windows 10/11, Linux, or macOS (Apple Silicon)

### Optional
- [LM Studio](https://lmstudio.ai/) for LLM post-processing

## Installation

### Quick Install (Windows)

```batch
:: Clone or download the repository
git clone https://github.com/yourusername/ClarityOCR.git
cd ClarityOCR

:: Run the installer (auto-detects Python 3.10-3.12)
install.bat
```

### Manual Install

#### For NVIDIA GPUs (Linux/Windows)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ClarityOCR
pip install -e .
```

#### For Apple Silicon (M1/M2/M3)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies for Apple Silicon (includes PyTorch with MPS support)
pip install -r requirements-apple-silicon.txt

# Install ClarityOCR
pip install -e .
```

For detailed Apple Silicon setup, see [README-APPLE-SILICON.md](README-APPLE-SILICON.md).

## Usage

### Web UI (Recommended)

```bash
# Start the web server
clarityocr serve

# Open in browser
# http://127.0.0.1:8008
```

The Web UI provides:
- File browser for PDF selection
- Real-time conversion progress with GPU stats
- OCR output preview
- LLM polish interface with diff view

### Command Line

```bash
# Convert PDFs in a directory
clarityocr convert --input-dir ./pdfs --output-dir ./markdown

# Convert specific PDF files
clarityocr convert file1.pdf file2.pdf

# Polish Markdown with LLM (requires LM Studio)
clarityocr polish --file document.md

# Scan directory to see conversion status
clarityocr scan ./documents
```

### Python API

```python
import clarityocr

# Convert a single PDF
output_path = clarityocr.convert_pdf("document.pdf")

# Polish OCR text with LLM
polished = clarityocr.polish_text(raw_ocr_text)

# Start the web server programmatically
clarityocr.start_server(host="127.0.0.1", port=8008)
```

## Configuration

### GPU Batch Sizes

ClarityOCR uses fixed, conservative batch sizes optimized for 16GB VRAM:

| Setting | Default | Description |
|---------|---------|-------------|
| Layout Batch | 4 | Layout model batch size |
| Recognition Batch | 8 | Text recognition batch size |
| Detection Batch | 8 | Object detection batch size |

These settings prevent VRAM overflow on large documents (500+ pages).

### LLM Settings (for post-processing)

| Setting | Default | Description |
|---------|---------|-------------|
| Base URL | `http://localhost:1234/v1` | LM Studio API endpoint |
| Temperature | 0.1 | Low for deterministic corrections |
| Chunk Size | 800 tokens | Text chunk size for processing |
| Max Tokens | 4096 | Maximum output tokens per chunk |

**Recommended LM Studio model**: RuadaptQwen3-8B-Hybrid (or any instruction-tuned model with 16K+ context)

## Output Format

ClarityOCR produces clean Markdown with:

- **Page markers**: `[p:1]`, `[p:2]`, etc. for stable citation references
- **Heading structure**: Properly detected and formatted headers
- **Tables**: Converted to Markdown table syntax
- **Images**: Extracted and saved alongside Markdown (optional)
- **Unicode**: Proper encoding with automatic mojibake repair

### Example Output

```markdown
[p:1]

# Chapter 1: Introduction

This is the introductory text of the document. The OCR accurately 
captures complex layouts and preserves the original formatting.

[p:2]

## 1.1 Background

| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

## LLM Post-Processing

The optional LLM polish step fixes common OCR errors:

### What It Fixes
- **Merged text**: `"Bibliography328"` → `"Bibliography 328"`
- **Similar characters**: `"сhapter"` (Cyrillic с) → `"chapter"` (Latin c)
- **Broken words**: `"trans-\nlation"` → `"translation"`
- **Spacing issues**: `"слово . слово"` → `"слово. слово"`
- **Typography**: `"..."` → `"…"`, `"-"` → `"—"` (em-dash)

### What It Preserves
- Page markers `[p:N]`
- Document structure
- Footnotes and citations
- Foreign language text

## Project Structure

```
ClarityOCR/
├── clarityocr/
│   ├── __init__.py      # Package API
│   ├── cli.py           # Command-line interface
│   ├── converter.py     # PDF conversion (marker-pdf wrapper)
│   ├── polish.py        # LLM post-processing
│   ├── server.py        # FastAPI web server
│   └── web/
│       └── static/      # HTML/CSS/JS for Web UI
├── requirements.txt
├── pyproject.toml
├── install.bat          # Windows installer
├── run.bat              # Quick start script
└── README.md
```

## Troubleshooting

### CUDA Out of Memory
If you encounter OOM errors on large PDFs:
1. Ensure no other GPU-intensive apps are running
2. The auto-fallback system will reduce batch sizes automatically
3. For very large PDFs (500+ pages), processing may use shared memory

### LM Studio Connection Failed
1. Ensure LM Studio is running
2. Start the local server (Server tab → Start Server)
3. Load a model before running polish
4. Check the server URL (default: `http://localhost:1234/v1`)

### Mojibake in Output
ClarityOCR automatically detects and fixes CP1251→Latin-1 encoding issues common in Russian PDFs. If you see garbled text, the automatic fix may not have triggered - check the source PDF encoding.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [marker-pdf](https://github.com/VikParuchuri/marker) - Core OCR engine
- [Surya](https://github.com/VikParuchuri/surya) - OCR models
- [LM Studio](https://lmstudio.ai/) - Local LLM inference
