from typing import Any, Dict, List, Tuple, Iterable
import csv

from .metrics import compute_em_and_cer
from .infer import run_text


def evaluate_text_only(model, tokenizer, dataset: List[Dict[str, Any]], indices: Iterable[int], instruction: str,
                        max_new_tokens: int = 128) -> Tuple[List[Dict[str, Any]], float, float]:
    rows: List[Dict[str, Any]] = []
    em_sum = 0
    cer_sum = 0.0
    n = 0
    for idx in indices:
        sample = dataset[idx]
        gt = sample["text"]
        pred = run_text(model, tokenizer, sample["image"], instruction, max_new_tokens=max_new_tokens)
        em, cer = compute_em_and_cer(pred, gt)
        rows.append({
            "id": sample.get("id", str(idx)),
            "type": sample.get("type"),
            "gt": gt,
            "pred": pred,
            "em": em,
            "cer": cer,
        })
        em_sum += em
        cer_sum += cer
        n += 1
    em_rate = em_sum / max(n, 1)
    cer_avg = cer_sum / max(n, 1)
    return rows, em_rate, cer_avg


def save_rows_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def compare_before_after(rows_base: List[Dict[str, Any]], rows_ft: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_id = {r["id"]: r for r in rows_base}
    ft_by_id = {r["id"]: r for r in rows_ft}
    joined: List[Dict[str, Any]] = []
    for sid, b in base_by_id.items():
        t = ft_by_id.get(sid)
        if not t:
            continue
        joined.append({
            "id": sid,
            "type": b.get("type"),
            "gt": b["gt"],
            "pred_base": b["pred"],
            "pred_ft": t["pred"],
            "em_base": b["em"],
            "em_ft": t["em"],
            "cer_base": b["cer"],
            "cer_ft": t["cer"],
            "cer_gain": b["cer"] - t["cer"],
        })
    return joined


