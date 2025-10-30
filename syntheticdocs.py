#!/usr/bin/env python3
import argparse
import os
import random
import uuid
import json
import pathlib
from datetime import datetime, timedelta
from typing import List, Callable, Dict, Any

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
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
])

def augment_image(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    arr2 = aug(image=arr)["image"]
    return Image.fromarray(arr2)

# ---------------------------------------------------------------------------
# Config + LLM helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(path_like: str) -> pathlib.Path:
    base = pathlib.Path(__file__).resolve().parent
    p = pathlib.Path(path_like)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p

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

def make_text_generator(mode: str, params: Dict[str, Any]) -> Callable[[str], str]:
    mode = (mode or "vllm").lower()
    max_tokens = int(params.get("max_tokens", 512))
    temperature = float(params.get("temperature", 0.8))

    if mode == "vllm":
        model_id = params.get("model")
        if not model_id:
            raise ValueError("For llm.mode='vllm' требуется указать 'model'")
        llm = prepare_llm(model_id, gpu_memory_util=float(params.get("gpu_memory_util", 0.90)))
        sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)

        def _gen(prompt: str) -> str:
            out = llm.generate([prompt], sampling)
            return out[0].outputs[0].text.strip()

        return _gen

    if mode == "openai":
        # Lazy import to avoid hard dependency at runtime if not used
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Требуется пакет 'openai' для режима openai. Установите его.") from e

        base_url = params.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = params.get("api_key")
        api_key_env = params.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key не найден (передайте 'api_key' или 'api_key_env' или переменную окружения OPENAI_API_KEY)")

        model_name = params.get("model")
        if not model_name:
            raise ValueError("Для llm.mode='openai' требуется указать 'model'")

        client = OpenAI(base_url=base_url, api_key=api_key)

        def _gen(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()

        return _gen

    raise ValueError(f"Неизвестный режим LLM: {mode}")

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
    gen_text_fn: Callable[[str], str],
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
        body   = gen_text_fn(prompt)

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
        "--config",
        help="Путь к JSON-конфигу",
    )
    parser.add_argument(
        "--llm-mode", choices=["vllm", "openai"], default=None,
        help="Режим LLM: локальная vLLM или OpenAI-совместимый API",
    )
    parser.add_argument(
        "-m", "--model",
        help="ID модели (HF repo-id/путь для vLLM или имя модели для OpenAI API)",
    )
    parser.add_argument(
        "--openai-base-url",
        help="Базовый URL OpenAI-совместимого API (например http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--openai-api-key",
        help="API-ключ для OpenAI-совместимого API",
    )
    parser.add_argument(
        "--openai-api-key-env",
        help="Имя переменной окружения с API-ключом",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Лимит токенов для LLM (перекрывает конфиг)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Температура выборки (перекрывает конфиг)",
    )
    parser.add_argument(
        "--font",
        help="Путь к TTF-шрифту с кириллицей (перекрывает конфиг)",
    )
    parser.add_argument(
        "--signatures",
        help="Директория с PNG-подписями (alpha, перекрывает конфиг)",
    )
    parser.add_argument(
        "--stamps",
        help="Директория с PNG-печатью (alpha, перекрывает конфиг)",
    )
    parser.add_argument(
        "-n", "--num", type=int,
        help="Количество документов (перекрывает конфиг)",
    )
    parser.add_argument(
        "-o", "--out",
        help="Папка для вывода (перекрывает конфиг)",
    )
    args = parser.parse_args()

    # Optional config loading
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)

    # Merge CLI over config
    llm_cfg = dict(cfg.get("llm", {}))
    if args.llm_mode is not None:
        llm_cfg["mode"] = args.llm_mode
    if args.model is not None:
        llm_cfg["model"] = args.model
    if args.openai_base_url is not None:
        llm_cfg["base_url"] = args.openai_base_url
    if args.openai_api_key is not None:
        llm_cfg["api_key"] = args.openai_api_key
    if args.openai_api_key_env is not None:
        llm_cfg["api_key_env"] = args.openai_api_key_env
    if args.max_tokens is not None:
        llm_cfg["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        llm_cfg["temperature"] = args.temperature

    # Generator config (paths, counts, seed)
    gen_cfg = dict(cfg.get("generator", {}))
    if args.font is not None:
        gen_cfg["font"] = args.font
    if args.signatures is not None:
        gen_cfg["signatures_dir"] = args.signatures
    if args.stamps is not None:
        gen_cfg["stamps_dir"] = args.stamps
    if args.out is not None:
        gen_cfg["out_dir"] = args.out
    if args.num is not None:
        gen_cfg["num"] = args.num

    required_gen_keys = ["font", "signatures_dir", "stamps_dir", "out_dir"]
    missing = [k for k in required_gen_keys if k not in gen_cfg or not gen_cfg[k]]
    if missing:
        raise SystemExit(
            "Отсутствуют обязательные параметры генератора: " + ", ".join(missing) +
            ". Укажите их в конфиге (generator.*) или через CLI."
        )

    # Seed (optional)
    if "seed" in gen_cfg and gen_cfg["seed"] is not None:
        try:
            seed_val = int(gen_cfg["seed"])
            random.seed(seed_val)
            np.random.seed(seed_val)
        except Exception:
            pass

    # Register the custom Cyrillic font
    font_path = resolve_path(gen_cfg["font"])
    pdfmetrics.registerFont(TTFont("CustomFont", str(font_path)))

    # Build text generator
    mode = llm_cfg.get("mode", "vllm")
    # If no config and mode is vLLM, ensure model provided
    if mode == "vllm" and not llm_cfg.get("model"):
        raise SystemExit("Укажите модель через --model или в конфиге (llm.model) для режима vllm")

    gen_text_fn = make_text_generator(mode, llm_cfg)

    generate_batch(
        gen_text_fn,
        int(gen_cfg.get("num", 1)),
        resolve_path(gen_cfg["signatures_dir"]),
        resolve_path(gen_cfg["stamps_dir"]),
        resolve_path(gen_cfg["out_dir"]),
    )

if __name__ == "__main__":
    main()
