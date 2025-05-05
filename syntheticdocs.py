#!/usr/bin/env python3
import argparse
import os
import random
import uuid
import json
import pathlib
from datetime import datetime, timedelta
from typing import List

from faker import Faker
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import albumentations as A

# ---------------------------------------------------------------------------
# Load prompt templates
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
with open(SCRIPT_DIR / "prompts.json", encoding="utf-8") as f:
    PROMPTS = json.load(f)["templates"]
DOC_TYPES = list(PROMPTS.keys())

# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------
aug = A.Compose([
    A.Perspective(scale=(0.02, 0.05), p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.5),
    A.ImageCompression(p=0.5),
    A.GaussNoise(sigma=(10.0, 50.0), p=0.5),
])

def augment_image(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    arr2 = aug(image=arr)["image"]
    return Image.fromarray(arr2)

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def prepare_llm(model_id: str, gpu_memory_util: float = 0.90) -> LLM:
    if os.path.isdir(model_id):
        local_dir = model_id
    else:
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=f"models/{model_id.replace('/', '_')}",
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    return LLM(model=local_dir, dtype="half", gpu_memory_utilization=gpu_memory_util)

def gen_text(llm: LLM, prompt: str, max_tokens: int = 512) -> str:
    out = llm.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.8))
    return out[0].outputs[0].text.strip()

# ---------------------------------------------------------------------------
# PDF rendering with signature & stamp
# ---------------------------------------------------------------------------
def draw_pdf(text: str, sig_png: str, stamp_png: str, out_pdf: str) -> (List[float], List[float]):
    w_pt, h_pt = A4
    margin = 40
    c = canvas.Canvas(out_pdf, pagesize=A4)

    # Draw body text
    txt = c.beginText(margin, h_pt - margin)
    txt.setFont("CustomFont", 12)
    for line in text.splitlines():
        txt.textLine(line if line.strip() else " ")
    c.drawText(txt)

    # Load overlays
    sig_img = Image.open(sig_png); sw_px, sh_px = sig_img.size
    st_img  = Image.open(stamp_png); tw_px, th_px = st_img.size

    # Convert pixels → points (300 dpi)
    DPI = 300
    sw_pt, sh_pt = sw_px * 72 / DPI, sh_px * 72 / DPI
    tw_pt, th_pt = tw_px * 72 / DPI, th_px * 72 / DPI

    # Scale overlays to fit page
    max_sw, max_sh = w_pt * 0.25, h_pt * 0.20
    max_tw, max_th = w_pt * 0.30, h_pt * 0.30
    scale_s = min(1, max_sw / sw_pt, max_sh / sh_pt)
    sw_pt, sh_pt = sw_pt * scale_s, sh_pt * scale_s
    scale_t = min(1, max_tw / tw_pt, max_th / th_pt)
    tw_pt, th_pt = tw_pt * scale_t, th_pt * scale_t

    # Random positions
    sig_x = random.uniform(margin, w_pt - margin - sw_pt)
    sig_y = random.uniform(margin, h_pt * 0.25 - sh_pt)
    st_x  = random.uniform(margin, w_pt * 0.35 - tw_pt)
    st_y  = random.uniform(margin, h_pt * 0.35 - th_pt)

    # Draw with transparency
    c.drawImage(ImageReader(sig_png), sig_x, sig_y, sw_pt, sh_pt, mask="auto")
    c.drawImage(ImageReader(stamp_png), st_x, st_y, tw_pt, th_pt, mask="auto")

    # Compute bboxes
    sig_bbox   = [sig_x, sig_y, sig_x + sw_pt, sig_y + sh_pt]
    stamp_bbox = [st_x, st_y, st_x + tw_pt, st_y + th_pt]

    c.showPage()
    c.save()
    return sig_bbox, stamp_bbox

# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate_batch(
    llm: LLM,
    n: int,
    sig_dir: pathlib.Path,
    stamp_dir: pathlib.Path,
    out_dir: pathlib.Path,
):
    fake   = Faker("ru_RU")
    sigs   = list(sig_dir.glob("*.png"))
    stamps = list(stamp_dir.glob("*.png"))
    if not sigs or not stamps:
        raise RuntimeError("PNG-активы подписей/штампов не найдены!")

    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        # Select document type and template
        doc_type = random.choice(DOC_TYPES)
        tpl      = PROMPTS[doc_type]

        # Generate fields
        dt0 = datetime.today() - timedelta(days=random.randint(10, 60))
        dt1 = dt0 + timedelta(days=random.randint(1, 10))
        contract_dt = (dt0 - timedelta(days=random.randint(30, 365))).strftime("%d.%m.%Y")

        fields = {
            "number": f"{random.randint(1,99):02d}/{random.randint(1,999):03d}/"
                      f"{random.randint(1,99):02d}/{random.randint(0,99):02d}",
            "status": random.choice(["открыто", "закрыто"]),
            "organization": fake.company(),
            "department": random.choice([
                "Контроль и диагностика в транспортных системах (авиация, автомобильные и железнодорожные)",
                "Финансовый отдел",
                "HR-отдел",
                "Юридический отдел"
            ]),
            "date_receipt":  dt0.strftime("%d.%m.%Y"),
            "date_dispatch": dt1.strftime("%d.%m.%Y"),
            "address_sender":   fake.address().replace("\n", ", "),
            "address_receiver": fake.address().replace("\n", ", "),
            "name_sender":      fake.name(),
            "name_receiver":    fake.name(),
            "subject":          random.choice([
                "Приемка оборудования",
                "Акт сверки",
                "Согласование договора",
                "Запрос информации",
                "Коммерческое предложение"
            ]),
            "content_instructions": (
                f"Согласно договору № {random.randint(1,999):03d} от {contract_dt}, оборудование "
                "должно прибыть в срок."
            ),
            "additional": "Добавить в конце «Резервная копия» и подготовить дополнительный штамп."
        }

        # Build prompt & generate
        prompt = tpl.format(**fields)
        body   = gen_text(llm, prompt)

        # Paths
        uid    = uuid.uuid4().hex
        pdf_p  = out_dir / f"{uid}.pdf"
        png_p  = out_dir / f"{uid}.png"
        json_p = out_dir / f"{uid}.json"

        # Render PDF + get bboxes
        sig_bbox, stamp_bbox = draw_pdf(
            body,
            str(random.choice(sigs)),
            str(random.choice(stamps)),
            str(pdf_p)
        )
        # Convert → PNG + augment
        img = convert_from_path(str(pdf_p), dpi=300, first_page=1, last_page=1)[0]
        img = augment_image(img)
        img.save(png_p, "PNG")

        # Save metadata
        meta = {
            "id":    uid,
            "type":  doc_type,
            "text":  body,
            "bboxes": [
                {"type": "signature", "coords": sig_bbox},
                {"type": "stamp",     "coords": stamp_bbox},
            ],
        }
        json_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        print(f"✔ [{i+1}/{n}] {doc_type} → {uid}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SyntheticDocs: генерирует PDF, PNG и JSON с метаданными",
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="HF repo-id или путь к локальной модели",
    )
    parser.add_argument(
        "--font", required=True,
        help="Путь к TTF-шрифту с кириллицей",
    )
    parser.add_argument(
        "--signatures", required=True,
        help="Директория с PNG-подписями (alpha)",
    )
    parser.add_argument(
        "--stamps", required=True,
        help="Директория с PNG-печатью (alpha)",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=1,
        help="Количество документов",
    )
    parser.add_argument(
        "-o", "--out", default="out",
        help="Папка для вывода",
    )
    args = parser.parse_args()

    # Register the custom Cyrillic font
    pdfmetrics.registerFont(TTFont("CustomFont", args.font))

    llm = prepare_llm(args.model)
    generate_batch(
        llm,
        args.num,
        pathlib.Path(args.signatures),
        pathlib.Path(args.stamps),
        pathlib.Path(args.out),
    )

if __name__ == "__main__":
    main()
