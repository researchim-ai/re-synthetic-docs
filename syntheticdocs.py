#!/usr/bin/env python3
import argparse
import os
import random
import uuid
import json
import pathlib
from datetime import datetime, timedelta
from typing import List, Callable, Dict, Any, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
# Augmentation pipeline (configurable)
# ---------------------------------------------------------------------------
def build_augmentation(cfg: Dict[str, Any] | None) -> Any:
    cfg = cfg or {}
    # Defaults
    persp = cfg.get("perspective", {"p": 0.5, "scale": [0.02, 0.05]})
    blur  = cfg.get("gaussian_blur", {"p": 0.5, "blur_limit": [1, 3]})
    comp  = cfg.get("compression", {"p": 0.5, "quality_range": [70, 100]})
    noise = cfg.get("gauss_noise", {"p": 0.5, "var_limit": [10.0, 50.0]})
    bc    = cfg.get("brightness_contrast", {"p": 0.4, "brightness_limit": 0.15, "contrast_limit": 0.15})

    transforms = []
    if (persp.get("p", 0) > 0):
        transforms.append(A.Perspective(scale=tuple(persp.get("scale", [0.02, 0.05])), p=float(persp.get("p", 0.5))))
    if (blur.get("p", 0) > 0):
        transforms.append(A.GaussianBlur(blur_limit=tuple(blur.get("blur_limit", [1, 3])), p=float(blur.get("p", 0.5))))
    if (comp.get("p", 0) > 0):
        # Albumentations будет варьировать качество сама; параметр quality_range неявный, используем p как вероятность
        transforms.append(A.ImageCompression(p=float(comp.get("p", 0.5))))
    if (noise.get("p", 0) > 0):
        # В некоторых версиях Albumentations параметр var_limit для GaussNoise помечается как устаревший/неподдерживаемый.
        # Используем ISONoise как более стабильную альтернативу для имитации скан-шума.
        transforms.append(A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.05, 0.5),
            p=float(noise.get("p", 0.5)),
        ))
    if (bc.get("p", 0) > 0):
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=float(bc.get("brightness_limit", 0.15)),
            contrast_limit=float(bc.get("contrast_limit", 0.15)),
            p=float(bc.get("p", 0.4)),
        ))
    if not transforms:
        return None
    # Включаем поддержку bbox (pascal_voc: [x1,y1,x2,y2])
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"], min_visibility=0.0),
    )

def augment_image(
    img: Image.Image,
    aug_pipeline: Any | None,
    bboxes: List[List[float]] | None = None,
    bbox_labels: List[str] | None = None,
) -> tuple[Image.Image, List[List[float]] | None, List[str] | None]:
    if aug_pipeline is None:
        return img, bboxes, bbox_labels
    arr = np.array(img)
    if bboxes is None:
        out = aug_pipeline(image=arr)
        return Image.fromarray(out["image"]), None, None
    out = aug_pipeline(image=arr, bboxes=bboxes, bbox_labels=bbox_labels or [])
    return Image.fromarray(out["image"]), out.get("bboxes", []), out.get("bbox_labels", bbox_labels)

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

        def _gen_batch(prompts: List[str]) -> List[str]:
            if not prompts:
                return []
            outs = llm.generate(prompts, sampling)
            return [o.outputs[0].text.strip() for o in outs]

        # прикрепляем батчевую версию как атрибут
        _gen.batch = _gen_batch  # type: ignore[attr-defined]
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

        def _gen_batch(prompts: List[str]) -> List[str]:
            # Параллелим запросы к локальному OpenAI-совместимому серверу
            max_workers = int(params.get("concurrency", 4))
            if not prompts:
                return []
            results: List[str] = [""] * len(prompts)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_gen, p): i for i, p in enumerate(prompts)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception:
                        results[idx] = ""
            return results

        _gen.batch = _gen_batch  # type: ignore[attr-defined]
        return _gen

    raise ValueError(f"Неизвестный режим LLM: {mode}")

# ---------------------------------------------------------------------------
# PDF rendering with signature & stamp
# ---------------------------------------------------------------------------
def draw_pdf(text: str, sig_png: str, stamp_png: str, out_pdf: str, layout_cfg: Dict[str, Any] | None = None) -> (List[float], List[float]):
    w_pt, h_pt = A4
    layout_cfg = layout_cfg or {}
    margin_range = layout_cfg.get("margin_range", [30, 60])
    font_size_range = layout_cfg.get("font_size_range", [11, 14])
    leading_multiplier_range = layout_cfg.get("leading_multiplier_range", [1.1, 1.4])

    margin = random.uniform(float(margin_range[0]), float(margin_range[1]))
    font_size = random.uniform(float(font_size_range[0]), float(font_size_range[1]))
    leading_mult = random.uniform(float(leading_multiplier_range[0]), float(leading_multiplier_range[1]))
    c = canvas.Canvas(out_pdf, pagesize=A4)

    # Draw body text
    txt = c.beginText(margin, h_pt - margin)
    txt.setFont("CustomFont", font_size)
    txt.setLeading(font_size * leading_mult)
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

    # Scale overlays to fit page (allow random scale within range)
    sig_scale_range = (layout_cfg.get("signature_scale_range", [0.15, 0.30]))
    stp_scale_range = (layout_cfg.get("stamp_scale_range", [0.20, 0.35]))
    max_sw, max_sh = w_pt * float(sig_scale_range[1]), h_pt * float(sig_scale_range[1])
    max_tw, max_th = w_pt * float(stp_scale_range[1]), h_pt * float(stp_scale_range[1])
    scale_s = min(1, max_sw / sw_pt, max_sh / sh_pt)
    sw_pt, sh_pt = sw_pt * scale_s, sh_pt * scale_s
    scale_t = min(1, max_tw / tw_pt, max_th / th_pt)
    tw_pt, th_pt = tw_pt * scale_t, th_pt * scale_t

    # Random positions (allow full-page ranges with inner margins)
    pos_cfg = layout_cfg.get("positions", {})
    sig_area = pos_cfg.get("signature_area", [0.05, 0.05, 0.95, 0.35])  # x1,y1,x2,y2 in [0..1]
    st_area  = pos_cfg.get("stamp_area",     [0.05, 0.05, 0.45, 0.45])
    sig_x = random.uniform(margin + sig_area[0]* (w_pt - 2*margin), margin + sig_area[2]* (w_pt - 2*margin) - sw_pt)
    sig_y = random.uniform(margin + sig_area[1]* (h_pt - 2*margin), margin + sig_area[3]* (h_pt - 2*margin) - sh_pt)
    st_x  = random.uniform(margin + st_area[0]* (w_pt - 2*margin),  margin + st_area[2]* (w_pt - 2*margin) - tw_pt)
    st_y  = random.uniform(margin + st_area[1]* (h_pt - 2*margin),  margin + st_area[3]* (h_pt - 2*margin) - th_pt)

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
def _render_worker_task(args: Dict[str, Any]) -> Dict[str, Any]:
    # args: text, sigs, stamps, pdf_p, png_p, json_p, layout_cfg, augment_cfg, doc_type, uid
    text: str = args["text"]
    sigs: List[str] = args["sigs"]
    stamps: List[str] = args["stamps"]
    pdf_p: pathlib.Path = pathlib.Path(args["pdf_p"])  # type: ignore
    png_p: pathlib.Path = pathlib.Path(args["png_p"])  # type: ignore
    json_p: pathlib.Path = pathlib.Path(args["json_p"])  # type: ignore
    layout_cfg: Dict[str, Any] | None = args.get("layout_cfg")
    augment_cfg: Dict[str, Any] | None = args.get("augment_cfg")
    doc_type: str = args.get("doc_type", "unknown")
    uid: str = args["uid"]

    sig_bbox, stamp_bbox = draw_pdf(
        text,
        str(random.choice(sigs)),
        str(random.choice(stamps)),
        str(pdf_p),
        layout_cfg=layout_cfg,
    )
    # Конвертируем PDF→PNG и масштабируем bbox в пиксели
    img = convert_from_path(str(pdf_p), dpi=300, first_page=1, last_page=1)[0]
    w_pt, h_pt = A4
    scale_x = img.width / float(w_pt)
    scale_y = img.height / float(h_pt)
    bboxes_px = [
        [sig_bbox[0] * scale_x, sig_bbox[1] * scale_y, sig_bbox[2] * scale_x, sig_bbox[3] * scale_y],
        [stamp_bbox[0] * scale_x, stamp_bbox[1] * scale_y, stamp_bbox[2] * scale_x, stamp_bbox[3] * scale_y],
    ]
    labels = ["signature", "stamp"]
    # Применяем аугментации с трансформацией bbox
    aug_pipeline = build_augmentation(augment_cfg)
    img, bboxes_px_out, labels_out = augment_image(img, aug_pipeline, bboxes_px, labels)
    img.save(png_p, "PNG")

    # Если аугментаций не было, используем исходные ббоксы в пикселях
    final_bboxes = bboxes_px_out if bboxes_px_out is not None else bboxes_px
    final_labels = labels_out if labels_out is not None else labels

    # Соберём JSON в исходном формате [{type, coords}]
    bboxes_json = []
    for lab, bb in zip(final_labels, final_bboxes):
        bboxes_json.append({"type": lab, "coords": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]})

    meta = {
        "id":    uid,
        "type":  doc_type,
        "text":  text,
        "bboxes": bboxes_json,
    }
    json_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {"uid": uid, "doc_type": doc_type}


def generate_batch(
    gen_text_fn: Callable[[str], str],
    n: int,
    sig_dir: pathlib.Path,
    stamp_dir: pathlib.Path,
    out_dir: pathlib.Path,
    augment_cfg: Dict[str, Any] | None = None,
    layout_cfg: Dict[str, Any] | None = None,
    batch_size: int = 8,
    render_workers: int | None = None,
):
    fake   = Faker("ru_RU")
    sigs   = list(sig_dir.glob("*.png"))
    stamps = list(stamp_dir.glob("*.png"))
    if not sigs or not stamps:
        raise RuntimeError("PNG-активы подписей/штампов не найдены!")

    out_dir.mkdir(parents=True, exist_ok=True)

    aug_pipeline = build_augmentation(augment_cfg)

    i = 0
    # Пул для рендера (CPU)
    use_pool = render_workers is not None and int(render_workers) > 0
    pool: ProcessPoolExecutor | None = None
    if use_pool:
        pool = ProcessPoolExecutor(max_workers=int(render_workers))
    pending = []  # список future для рендеринга
    while i < n:
        # подготовим батч промптов и заготовок путей
        batch_prompts: List[str] = []
        batch_meta: List[Dict[str, Any]] = []
        take = min(batch_size, n - i)
        for _ in range(take):
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

            prompt = tpl.format(**fields)
            uid    = uuid.uuid4().hex
            pdf_p  = out_dir / f"{uid}.pdf"
            png_p  = out_dir / f"{uid}.png"
            json_p = out_dir / f"{uid}.json"

            batch_prompts.append(prompt)
            batch_meta.append({
                "uid": uid,
                "doc_type": doc_type,
                "pdf_p": pdf_p,
                "png_p": png_p,
                "json_p": json_p,
            })

        # сгенерируем батч текстов
        if hasattr(gen_text_fn, "batch"):
            bodies: List[str] = gen_text_fn.batch(batch_prompts)  # type: ignore[attr-defined]
        else:
            bodies = [gen_text_fn(p) for p in batch_prompts]

        # отрисуем и сохраним результаты
        for k, body in enumerate(bodies):
            meta_k = batch_meta[k]
            task_args = {
                "text": body,
                "sigs": [str(p) for p in sigs],
                "stamps": [str(p) for p in stamps],
                "pdf_p": str(meta_k["pdf_p"]),
                "png_p": str(meta_k["png_p"]),
                "json_p": str(meta_k["json_p"]),
                "layout_cfg": layout_cfg,
                "augment_cfg": augment_cfg,
                "doc_type": meta_k["doc_type"],
                "uid": meta_k["uid"],
            }
            if pool is not None:
                pending.append(pool.submit(_render_worker_task, task_args))
            else:
                res = _render_worker_task(task_args)
                print(f"✔ [{i+1}/{n}] {res['doc_type']} → {res['uid']}")
                i += 1

        # если используем пул — частично дожидаемся результатов, чтобы поддерживать прогресс
        if pool is not None:
            # выгружаем не менее одной партии результатов
            done_any = 0
            new_pending = []
            for fut in pending:
                if fut.done():
                    try:
                        res = fut.result()
                        print(f"✔ [{i+1}/{n}] {res['doc_type']} → {res['uid']}")
                    except Exception:
                        pass
                    i += 1
                    done_any += 1
                else:
                    new_pending.append(fut)
            pending = new_pending

    # Дождаться оставшихся задач пула
    if pool is not None:
        for fut in pending:
            try:
                res = fut.result()
                print(f"✔ [{n}/{n}] {res['doc_type']} → {res['uid']}")
            except Exception:
                pass
        pool.shutdown(wait=True)

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
        augment_cfg=cfg.get("augment"),
        layout_cfg=cfg.get("layout"),
    )

if __name__ == "__main__":
    main()
