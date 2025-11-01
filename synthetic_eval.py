import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Text normalization and metrics
# -----------------------------


def normalize_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return float(len(pred) > 0)
    return levenshtein_distance(pred, gt) / len(gt)


# -----------------------------
# Prompts
# -----------------------------


def get_ocr_text_prompt() -> str:
    return (
        "Внимательно распознай весь печатный текст на изображении. "
        "Выведи ТОЛЬКО распознанный текст без каких-либо комментариев, разметки, префиксов и кавычек. "
        "Сохраняй порядок строк и знаки препинания. Если символ неразборчив, не выдумывай его."
    )


def get_ocr_text_bbox_prompt() -> str:
    return (
        "Распознай текст на изображении и координаты важных областей (подпись и печать). "
        "Ответ верни строго в формате JSON без пояснений: "
        "{\n"
        "  \"text\": \"ПОЛНЫЙ_ТЕКСТ\",\n"
        "  \"bboxes\": [\n"
        "    {\"type\": \"signature\", \"coords\": [x1, y1, x2, y2]},\n"
        "    {\"type\": \"stamp\",     \"coords\": [x1, y1, x2, y2]}\n"
        "  ]\n"
        "}\n"
        "Где coords — пиксельные координаты прямоугольника: левый-верхний (x1,y1) и правый-нижний (x2,y2)."
    )


# -----------------------------
# Input building for VLM
# -----------------------------


def build_multimodal_inputs(tokenizer, image, instruction: str):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return inputs


@dataclass
class DecodeParams:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


def generate_text(
    model,
    tokenizer,
    inputs,
    decode: Optional[DecodeParams] = None,
) -> str:
    from transformers import TextStreamer

    params = decode or DecodeParams()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=params.max_new_tokens,
        use_cache=True,
        temperature=params.temperature,
        top_p=params.top_p,
        do_sample=params.do_sample,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalize_text(decoded)


# -----------------------------
# BBox parsing and IoU
# -----------------------------


def try_extract_json(payload: str) -> Optional[Dict[str, Any]]:
    s = payload.strip()
    # Remove code fences if any
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # Find the first {...} JSON block if extra text present
    m = re.search(r"\{[\s\S]*\}$", s)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-9
    return inter / union


def extract_bboxes_from_json(pred_json_text: str) -> Dict[str, List[List[float]]]:
    obj = try_extract_json(pred_json_text) or {}
    result: Dict[str, List[List[float]]] = {}
    for item in obj.get("bboxes", []) or []:
        typ = str(item.get("type") or "unknown").strip()
        coords = item.get("coords") or []
        if (
            isinstance(coords, list)
            and len(coords) == 4
            and all(isinstance(x, (int, float)) for x in coords)
        ):
            result.setdefault(typ, []).append([float(c) for c in coords])
    return result


def compute_iou_by_type(
    gt_bboxes: List[Dict[str, Any]],
    pr_bboxes: Dict[str, List[List[float]]],
) -> Dict[str, float]:
    # Greedy 1-1 matching by IoU for each type
    from math import isfinite

    gt_by_type: Dict[str, List[List[float]]] = {}
    for b in gt_bboxes or []:
        typ = str(b.get("type") or "unknown").strip()
        coords = b.get("coords") or []
        if (
            isinstance(coords, list)
            and len(coords) == 4
            and all(isinstance(x, (int, float)) for x in coords)
        ):
            gt_by_type.setdefault(typ, []).append([float(c) for c in coords])

    scores: Dict[str, float] = {}
    for typ, gts in gt_by_type.items():
        prs = [c for c in pr_bboxes.get(typ, [])]
        used = [False] * len(prs)
        ious: List[float] = []
        for g in gts:
            best_iou = 0.0
            best_j = -1
            for j, p in enumerate(prs):
                if used[j]:
                    continue
                val = bbox_iou(g, p)
                if val > best_iou:
                    best_iou = val
                    best_j = j
            if best_j >= 0:
                used[best_j] = True
                ious.append(best_iou)
            else:
                ious.append(0.0)
        if ious:
            scores[typ] = sum(ious) / len(ious)
    return scores


# -----------------------------
# Evaluation runners
# -----------------------------


def eval_text_on_dataset(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    indices: List[int],
    instruction: str,
    decode: Optional[DecodeParams] = None,
) -> Tuple[List[Dict[str, Any]], float, float]:
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    rows: List[Dict[str, Any]] = []
    exact_matches = 0
    cers: List[float] = []
    for idx in indices:
        sample = dataset[idx]
        gt = normalize_text(sample["text"])
        inputs = build_multimodal_inputs(tokenizer, sample["image"], instruction)
        pred = generate_text(model, tokenizer, inputs, decode)
        em = int(pred == gt)
        exact_matches += em
        c = cer(pred, gt)
        cers.append(c)
        rows.append({
            "id": sample.get("id", str(idx)),
            "type": sample.get("type"),
            "gt": gt,
            "pred": pred,
            "em": em,
            "cer": c,
        })
    em_rate = exact_matches / len(indices)
    avg_cer = sum(cers) / len(indices)
    return rows, em_rate, avg_cer


def eval_text_bbox_on_dataset(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    indices: List[int],
    instruction: str,
    decode: Optional[DecodeParams] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    rows: List[Dict[str, Any]] = []
    iou_by_type_acc: Dict[str, List[float]] = {}
    for idx in indices:
        sample = dataset[idx]
        inputs = build_multimodal_inputs(tokenizer, sample["image"], instruction)
        pred_json = generate_text(model, tokenizer, inputs, decode)
        pr_bboxes = extract_bboxes_from_json(pred_json)
        gt_bboxes = sample.get("bboxes") or []
        iou_types = compute_iou_by_type(gt_bboxes, pr_bboxes)
        for t, v in iou_types.items():
            iou_by_type_acc.setdefault(t, []).append(float(v))
        rows.append({
            "id": sample.get("id", str(idx)),
            "type": sample.get("type"),
            "pred_json": pred_json,
            "iou": iou_types,
        })
    iou_summary: Dict[str, float] = {k: (sum(v) / len(v) if v else 0.0) for k, v in iou_by_type_acc.items()}
    return rows, iou_summary


# -----------------------------
# Dataset helper
# -----------------------------


def load_syntheticdocs(out_dir: str, pil_loader) -> List[Dict[str, Any]]:
    import glob
    import os

    data: List[Dict[str, Any]] = []
    for jpath in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            text = (meta.get("text") or "").strip()
            if not text:
                continue
            stem = os.path.splitext(os.path.basename(jpath))[0]
            ppath = os.path.join(out_dir, stem + ".png")
            if not os.path.exists(ppath):
                continue
            img = pil_loader(ppath)
            data.append({
                "image": img,
                "text": text,
                "id": meta.get("id") or stem,
                "type": meta.get("type"),
                "bboxes": meta.get("bboxes") or [],
            })
        except Exception:
            continue
    return data


