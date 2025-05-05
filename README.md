Thought for a few seconds


````markdown
# SyntheticDocs

**An end-to-end toolkit for generating synthetic “scanned” A4 documents with signatures, stamps, and scanner artifacts, for VLM/LLM training.**

> ❗️ **Research-only:** This project is intended **exclusively** for research and evaluation. Generated documents **must not** be used in any real-world or production scenarios, nor for deceptive or fraudulent activities.

---

## Features

- **LLM-driven text** via local vLLM + Hugging Face models  
- **Template-based prompts** loaded from `prompts.json` for multiple document types  
- **Cyrillic font support** for realistic Russian text in PDFs  
- **Signature & stamp overlays** with automatic scaling and alpha transparency  
- **Scanner artifacts** (perspective warp, blur, JPEG-style compression, noise) via Albumentations  
- **Outputs**:  
  - `.pdf` — original document  
  - `.png` — high-res “scan” at 300 DPI  
  - `.json` — metadata (bounding boxes + raw LLM text)  

---

## Project Structure

```text
re-synthetic-docs/
├── assets/
│   ├── signatures/
│   │   ├── fonts/        # .ttf fonts for signature generation
│   │   └── png/          # signature PNGs (alpha channel)
│   └── stamps/
│       ├── fonts/        # .ttf fonts for stamp generation
│       └── png/          # stamp PNGs (alpha channel)
├── prompts.json          # JSON templates for each document type
├── make_signatures.py    # script to generate signature PNGs from fonts
├── make_stamps.py        # script to generate circular stamp PNGs
├── syntheticdocs.py      # main generator: LLM → PDF → PNG → metadata
└── requirements.txt      # Python dependencies
````

---

## Installation

```bash
git clone https://github.com/your-org/re-synthetic-docs.git
cd re-synthetic-docs

# (optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Notes:**
>
> * Requires **Python 3.10+**.
> * For GPU acceleration, install a compatible `torch` + CUDA build **before** installing `vllm`.
> * On Linux, you may need `poppler-utils` for PDF→PNG conversion:
>
>   ```bash
>   sudo apt update && sudo apt install -y poppler-utils
>   ```

---

## Preparing Assets

1. **Fonts**
   Place at least one Cyrillic-capable `.ttf` in
   `assets/signatures/fonts/` and `assets/stamps/fonts/`.

2. **Generate signatures** (optional)

   ```bash
   python make_signatures.py
   ```

   — will produce \~500 PNG signatures in `assets/signatures/png/`.

3. **Generate stamps** (optional)

   ```bash
   python make_stamps.py
   ```

   — will produce \~30 PNG stamps in `assets/stamps/png/`.

---

## Usage

```bash
python syntheticdocs.py \
  --model     "/path/to/model-or-repo-id" \
  --font      assets/signatures/fonts/YourCyrillicFont.ttf \
  --signatures assets/signatures/png \
  --stamps    assets/stamps/png \
  --out       out \
  -n          10
```

* **`--model`**      — Hugging Face repo-ID (e.g. `mistralai/Mistral-7B-Instruct-v0.2`) or local model directory
* **`--font`**       — path to a `.ttf` font with Cyrillic support
* **`--signatures`** — folder containing signature PNGs (alpha channel)
* **`--stamps`**     — folder containing stamp PNGs (alpha channel)
* **`--out`**        — output directory (created if missing)
* **`-n`**           — number of documents to generate

After running, `out/` will contain:

```text
out/
├── <uuid1>.pdf
├── <uuid1>.png
├── <uuid1>.json
├── <uuid2>.pdf
├── <uuid2>.png
├── <uuid2>.json
└── …
```

* `<uuid>.pdf`  — generated PDF
* `<uuid>.png`  — synthetic “scan” (300 DPI)
* `<uuid>.json` — metadata:

  ```jsonc
  {
    "id": "<uuid>",
    "type": "<document_type>",
    "text": "<raw LLM-generated text>",
    "bboxes": [
      { "type": "signature", "coords": [x0, y0, x1, y1] },
      { "type": "stamp",     "coords": [x0, y0, x1, y1] }
    ]
  }
  ```

---

## Customization & Extension

* **Add document types**: extend `prompts.json` with new templates.
* **Adjust scanner effects**: modify the Albumentations pipeline (`aug = A.Compose([...])`).
* **Multi-page documents**: enhance `draw_pdf()` to add additional pages or fields.
* **Parallel generation**: batch LLM requests or use `asyncio` for speed.

---

## License & Disclaimer

This code is provided “as is” for **research purposes only**.
The authors disclaim any liability for misuse.

```
```
