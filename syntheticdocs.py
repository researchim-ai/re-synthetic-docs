# syntheticdocs.py – minimal end‑to‑end generator (LLM→PDF→PNG)
"""
Usage
-----
$ python syntheticdocs.py --model mistralai/Mistral-7B-Instruct-v0.2 \
                         --signatures assets/signatures \
                         --stamps assets/stamps \
                         --out out_dir -n 5

The script will
1. download / cache the HF model into ./models/…
2. launch vLLM for local inference
3. generate n synthetic Russian documents (topic chosen at random)
4. embed a random signature + stamp (PNG with alpha) at random positions
5. save <uuid>.pdf, <uuid>.png (300 dpi) and <uuid>.json (raw text)

Dependencies (append to requirements.txt)
-----------------------------------------
vllm>=0.4.2
huggingface_hub>=0.23.0
reportlab>=4.0.8
pdf2image>=1.16.3
Pillow>=10.3.0

For GPU CUDA 12.1+ and Python 3.10/3.11.
Make sure you install the correct torch+cuda build before vllm.
"""

import argparse, os, random, uuid, json, pathlib, tempfile
from typing import List

from huggingface_hub import snapshot_download  # HF model fetcher
from vllm import LLM, SamplingParams            # local LLM runtime

from reportlab.lib import pagesizes
from reportlab.pdfgen import canvas
from PIL import Image
from pdf2image import convert_from_path


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def prepare_llm(model_id: str, gpu_memory_util: float = 0.90) -> LLM:
    """Download (if necessary) Hugging Face model + spin up vLLM instance."""
    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=f"models/{model_id.replace('/', '_')}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return LLM(model=local_dir, dtype="half", gpu_memory_utilization=gpu_memory_util)


def gen_text(llm: LLM, topic: str, max_tokens: int = 512) -> str:
    prompt = (
        f"Напиши официальное письмо или служебный документ (ок. 1 стр. A4) на тему:\n"
        f"— {topic}\nСтиль: деловой, канцелярский. Добавь реальные даты, числа, ФИО, адреса."
    )
    out = llm.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.8))
    return out[0].outputs[0].text.strip()


TOPICS: List[str] = [
    "акт сверки",
    "служебная записка",
    "договор аренды",
    "счёт‑фактура",
    "заявление на отпуск",
]

# ---------------------------------------------------------------------------
# PDF & graphics
# ---------------------------------------------------------------------------

def make_pdf(text: str, sig_png: str, stamp_png: str, out_pdf: str):
    w, h = pagesizes.A4  # points (1/72 inch)
    margin = 40
    c = canvas.Canvas(out_pdf, pagesize=pagesizes.A4)

    # body text
    txt = c.beginText(margin, h - margin)
    txt.setFont("Helvetica", 12)
    for line in text.splitlines():
        if not line.strip():
            line = " "  # preserve blank lines
        txt.textLine(line)
    c.drawText(txt)

    # load overlay assets (keep original size)
    sig_img = Image.open(sig_png)
    stamp_img = Image.open(stamp_png)

    # random placement heuristics
    sig_x = random.randint(int(w * 0.55), int(w * 0.85))
    sig_y = random.randint(int(h * 0.10), int(h * 0.25))
    st_x = random.randint(int(w * 0.15), int(w * 0.35))
    st_y = random.randint(int(h * 0.10), int(h * 0.35))

    c.drawInlineImage(sig_img, sig_x, sig_y)
    c.drawInlineImage(stamp_img, st_x, st_y)

    c.showPage()
    c.save()


def pdf_to_png(pdf_path: str, png_path: str, dpi: int = 300):
    img = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)[0]
    img.save(png_path, "PNG")


# ---------------------------------------------------------------------------
# Main generator loop
# ---------------------------------------------------------------------------

def generate_batch(llm: LLM, n: int, sig_dir: pathlib.Path, stamp_dir: pathlib.Path, out_dir: pathlib.Path):
    sig_list = list(sig_dir.glob("*.png"))
    st_list = list(stamp_dir.glob("*.png"))
    if not sig_list or not st_list:
        raise RuntimeError("Signature or stamp PNG assets not found!")

    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(n):
        doc_id = uuid.uuid4().hex
        topic = random.choice(TOPICS)
        body = gen_text(llm, topic)

        sig_path = random.choice(sig_list)
        st_path = random.choice(st_list)

        pdf_path = out_dir / f"{doc_id}.pdf"
        png_path = out_dir / f"{doc_id}.png"
        json_path = out_dir / f"{doc_id}.json"

        make_pdf(body, str(sig_path), str(st_path), str(pdf_path))
        pdf_to_png(str(pdf_path), str(png_path))

        meta = {"id": doc_id, "topic": topic, "text": body}
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print("✔", doc_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic document generator (LLM→PDF→PNG)")
    parser.add_argument("-m", "--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="HF model repo id")
    parser.add_argument("-n", "--num", type=int, default=1, help="How many docs to create")
    parser.add_argument("--signatures", type=pathlib.Path, required=True, help="Dir with signature PNGs (α‑channel)")
    parser.add_argument("--stamps", type=pathlib.Path, required=True, help="Dir with stamp PNGs (α‑channel)")
    parser.add_argument("-o", "--out", type=pathlib.Path, default="out", help="Output dir")
    args = parser.parse_args()

    llm = prepare_llm(args.model)
    generate_batch(llm, args.num, args.signatures, args.stamps, args.out)
