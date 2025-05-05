Thought for a couple of seconds


```markdown
# SyntheticDocs

**_An end-to-end toolkit for generating synthetic scanned A4 documents with signatures, stamps, and OCR-style noise for VLM/LLM training._**

> **❗️ Research‐only:** This project is intended _exclusively_ for research and evaluation purposes. Generated documents should **not** be used for any real-world or production scenarios, nor for deceptive or fraudulent activities.

---

## Features

- **LLM-driven text generation** via [vLLM](https://github.com/vllm-project/vllm) + Hugging Face models  
- **Template‐based prompts** loaded from `prompts.json` for multiple document types  
- **Custom Cyrillic font** support for realistic Russian text  
- **Signature & stamp overlays** with automatic scaling & transparency  
- **Scanner effects** (perspective warp, blur, JPEG artifacts, noise) via Albumentations  
- **Output:**  
  - Multi‐page PDF (`.pdf`)  
  - High‐res PNG scan (`.png`, 300 DPI)  
  - Metadata JSON (`.json`) with bounding boxes and raw LLM text  

---

## Repository Structure

```

re-synthetic-docs/
├── assets/
│   ├── signatures/
│   │   ├── fonts/               # .ttf fonts for signature generation
│   │   └── png/                 # existing signature PNGs
│   └── stamps/
│       ├── fonts/               # .ttf fonts for stamp generation
│       └── png/                 # existing stamp PNGs
├── prompts.json                # Prompt templates per document type
├── make\_signatures.py          # Script to generate signature PNGs via fonts
├── make\_stamps.py              # Script to generate circular stamp PNGs
├── syntheticdocs.py            # Main generator: LLM → PDF → PNG → metadata
└── requirements.txt            # Python dependencies

````

---

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/your-org/re-synthetic-docs.git
cd re-synthetic-docs

# Create & activate virtualenv (optional)
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
````

> **Note:**
>
> * Requires Python 3.10+
> * For GPU acceleration, install the appropriate `torch`+CUDA build before `vllm`.
> * `poppler-utils` (for `pdftoppm`) may be needed on Linux:
>
>   ```bash
>   sudo apt update && sudo apt install -y poppler-utils
>   ```

### 2. Prepare Assets

1. **Fonts**

   * Place at least one Cyrillic‐capable `.ttf` into `assets/signatures/fonts/` and `assets/stamps/fonts/`.

2. **Generate Signatures (optional)**

   ```bash
   python make_signatures.py
   ```

   * Produces \~500 `.png` signatures in `assets/signatures/png/`.

3. **Generate Stamps (optional)**

   ```bash
   python make_stamps.py
   ```

   * Produces \~30 circular stamp PNGs in `assets/stamps/png/`.

### 3. Configure Prompts

* Open `prompts.json` to review or customize prompt templates for each document type.
* Templates use Python-style `{field}` placeholders.

### 4. Run the Generator

```bash
python syntheticdocs.py \
  --model   "/path/to/your/hf-model-or-repo" \
  --font    assets/signatures/fonts/YourCyrillicFont.ttf \
  --signatures assets/signatures/png \
  --stamps  assets/stamps/png \
  --out     out \
  -n        10
```

* **`--model`**: HF repo-ID (e.g. `mistralai/Mistral-7B-Instruct-v0.2`) or local model directory
* **`--font`**: Path to your Cyrillic `.ttf` for PDF text
* **`--signatures`**, **`--stamps`**: Directories with PNG overlays (α-channel)
* **`--out`**: Output directory (will be created if missing)
* **`-n`**: Number of documents to generate

Each run produces:

```
out/
├── <uuid1>.pdf
├── <uuid1>.png
├── <uuid1>.json
├── <uuid2>.pdf
├── <uuid2>.png
├── <uuid2>.json
└── …
```

* **`<uuid>.pdf`**: Generated PDF
* **`<uuid>.png`**: Synthetic “scanned” image (300 DPI)
* **`<uuid>.json`**:

  ```jsonc
  {
    "id": "<uuid>",
    "type": "<doc_type>",
    "text": "<raw LLM-generated text>",
    "bboxes": [
      { "type": "signature", "coords": [x0, y0, x1, y1] },
      { "type": "stamp",     "coords": [x0, y0, x1, y1] }
    ]
  }
  ```

---

## Customization & Extension

* **Add document types**: Extend `prompts.json` with new keys & templates.
* **Adjust scanner effects**: Tweak `aug = A.Compose([...])` in `syntheticdocs.py`.
* **Multi-page support**: Modify `draw_pdf()` to add extra pages/fields.
* **Parallel generation**: Integrate `asyncio` or batch calls in `vLLM`.

---

## License & Disclaimer

This code is provided “as is” for **research purposes only**.
Use at your own risk. The authors are not responsible for any misuse.

---

Happy experimenting! 🚀
