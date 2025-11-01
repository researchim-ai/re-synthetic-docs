from typing import List, Dict, Any, Tuple


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_by_label(pred_bboxes: List[Dict[str, Any]], gt_bboxes: List[Dict[str, Any]], iou_thr: float = 0.5) -> Dict[str, Any]:
    """Простая метрика: для каждой метки (signature/stamp) ищем лучшее совпадение и считаем IoU."""
    res = {"signature": None, "stamp": None}
    for label in list(res.keys()):
        gt = [b for b in gt_bboxes if b.get("type") == label or b.get("label") == label]
        pr = [b for b in pred_bboxes if b.get("type") == label or b.get("label") == label]
        best = 0.0
        for g in gt:
            for p in pr:
                gi = g.get("coords") or g.get("bbox")
                pi = p.get("coords") or p.get("bbox")
                if not (isinstance(gi, list) and isinstance(pi, list) and len(gi) == 4 and len(pi) == 4):
                    continue
                best = max(best, iou_xyxy(gi, pi))
        res[label] = {"best_iou": best, "hit": best >= iou_thr}
    return res


