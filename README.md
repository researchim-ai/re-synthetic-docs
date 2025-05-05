# SyntheticDocs

**A data-generation toolkit for creating synthetic “scanned” A4 document datasets—complete with text, layout annotations, signatures and stamps—for training vision-language models (VLMs) on document understanding tasks.**

> ❗️ **Research-only:** This project exists solely to produce labeled synthetic data for VLM research. Generated images **must not** be used in any live/production system or for deceptive purposes.

---

## Purpose

Many VLM applications (document layout analysis, signature/stamp detection, OCR + grounding) suffer from a lack of large, richly annotated datasets. **SyntheticDocs** automatically produces:

- High-resolution “scanned” document images (`.png`, 300 DPI)  
- Matching original PDFs (`.pdf`)  
- Structured metadata (`.json`) containing:
  - **Text content** (raw LLM output)
  - **Bounding boxes** for each signature and stamp
  - **Document type** and prompt fields

These artifacts form end-to-end training samples for VLMs to learn both **visual layout** and **textual grounding**.

---

## Key Features

- **LLM-driven text**—vLLM + Hugging Face models generate realistic document bodies  
- **Prompt templates** in `prompts.json` cover multiple document genres (memos, invoices, contracts…)  
- **Cyrillic font support**—realistic Russian text rendering in PDFs  
- **Signature & stamp overlays**—auto-scaled, alpha-preserved PNG assets  
- **Scanner artifacts**—perspective warp, blur, JPEG compression, noise via Albumentations  
- **Rich metadata**—JSON files include text, document type, and precise bounding boxes

---

## Project Layout

```text
re-synthetic-docs/
├── assets/  
│   ├── signatures/  
│   │   ├── fonts/        # .ttf fonts for signature generation  
│   │   └── png/          # signature PNGs (alpha channel)  
│   └── stamps/  
│       ├── fonts/        # .ttf fonts for stamp generation  
│       └── png/          # stamp PNGs (alpha channel)  
├── prompts.json          # Prompt templates by document type  
├── make_signatures.py    # Generate signature PNGs from fonts  
├── make_stamps.py        # Generate stamp PNGs  
├── syntheticdocs.py      # Main generator: LLM → PDF → PNG → metadata  
└── requirements.txt      # Python dependencies  
