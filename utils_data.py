"""Data loading helpers for synthetic docs dataset.

Loads paired PNG + JSON with fields: id, type, text, and optional bboxes in the form
[{"type": "signature"|"stamp", "coords": [x1,y1,x2,y2]}].
"""

from __future__ import annotations

import os
import json
import glob
from typing import Any, Dict, List, Optional

from PIL import Image


def load_syntheticdocs(
    out_dir: str,
    max_samples: Optional[int] = None,
    max_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """Load synthetic document samples from an output directory.

    Returns list of dicts with keys: image (PIL.Image), text, id, type, bboxes (optional).
    """
    data: List[Dict[str, Any]] = []
    json_paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    for jpath in json_paths:
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            text = (meta.get("text") or "").strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            stem = os.path.splitext(os.path.basename(jpath))[0]
            ppath = os.path.join(out_dir, stem + ".png")
            if not os.path.exists(ppath):
                continue
            img = Image.open(ppath).convert("RGB")
            sample: Dict[str, Any] = {
                "image": img,
                "text": text,
                "id": meta.get("id", stem),
                "type": meta.get("type"),
            }
            if isinstance(meta.get("bboxes"), list):
                sample["bboxes"] = meta["bboxes"]
            data.append(sample)
            if max_samples and len(data) >= max_samples:
                break
        except Exception:
            # Skip malformed items silently for robustness in notebooks
            continue
    return data


