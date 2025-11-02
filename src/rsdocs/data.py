import os
import json
import glob
from typing import Any, Dict, List, Optional
from PIL import Image


def load_syntheticdocs(out_dir: str, max_samples: Optional[int] = None, max_chars: int = 4000) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for jpath in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
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
                "id": meta.get("id", stem),
                "type": meta.get("type"),
                "text": text,
                "image_path": ppath,
                "image": img,
            }
            if isinstance(meta.get("bboxes"), list):
                sample["bboxes"] = meta["bboxes"]
            data.append(sample)
            if max_samples and len(data) >= max_samples:
                break
        except Exception:
            continue
    return data


def load_split(root_dir: str, max_samples_train: Optional[int] = None, max_samples_test: Optional[int] = None,
               max_chars: int = 4000) -> Dict[str, List[Dict[str, Any]]]:
    """Загружает два набора: data/train и data/test внутри root_dir."""
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    train = load_syntheticdocs(train_dir, max_samples=max_samples_train, max_chars=max_chars) if os.path.isdir(train_dir) else []
    test = load_syntheticdocs(test_dir, max_samples=max_samples_test, max_chars=max_chars) if os.path.isdir(test_dir) else []
    return {"train": train, "test": test}


