import argparse
import json
import random
import uuid
import pathlib
from typing import Dict, Any
from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

# ------------------------------------------------------------
# Script: make_signatures.py
# Purpose: Generate handwritten-like signature PNGs with alpha channel
# Config: Read settings from JSON (--config), CLI overrides optional
# Usage: python make_signatures.py --config config.example.json
# ------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_paths(sig_cfg: Dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path]:
    script_dir = pathlib.Path(__file__).resolve().parent
    fonts_dir = pathlib.Path(sig_cfg.get("fonts_dir", script_dir / "assets" / "signatures" / "fonts"))
    if not fonts_dir.is_absolute():
        fonts_dir = (script_dir / fonts_dir).resolve()
    out_dir = pathlib.Path(sig_cfg.get("output_dir", script_dir / "assets" / "signatures" / "png"))
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return fonts_dir, out_dir

def supports_cyrillic(font_path: pathlib.Path) -> bool:
    try:
        ft = ImageFont.truetype(str(font_path), 64)
        mask = ft.getmask("Тест")
        return mask.getbbox() is not None
    except Exception:
        return False

def determine_locale(sig_cfg: Dict[str, Any], all_fonts: list[pathlib.Path]) -> str:
    desired = sig_cfg.get("locale", "auto")
    if desired != "auto":
        return desired
    cyrillic_fonts = [f for f in all_fonts if supports_cyrillic(f)]
    if cyrillic_fonts:
        print(f"[DEBUG] Cyrillic fonts: {[f.name for f in cyrillic_fonts]}")
        return "ru_RU"
    print("[WARN] No Cyrillic-supporting fonts detected. Falling back to English names for signatures.")
    return "en_US"

def choose_fonts(fonts_dir: pathlib.Path) -> list[pathlib.Path]:
    all_fonts = list(fonts_dir.glob("*.ttf"))
    print(f"[DEBUG] looking for fonts in: {fonts_dir}")
    print(f"[DEBUG] found fonts: {[f.name for f in all_fonts]}")
    if not all_fonts:
        print("[ERROR] No .ttf fonts found in signatures/fonts. Please add at least one font.")
        exit(1)
    return all_fonts

def render_signature(name: str, font_path: pathlib.Path, sig_cfg: Dict[str, Any]) -> Image.Image:
    fs_cfg = sig_cfg.get("font_size", {})
    font_size_min = int(fs_cfg.get("min", 48))
    font_size_max = int(fs_cfg.get("max", 72))
    font_size = random.randint(font_size_min, font_size_max)
    font = ImageFont.truetype(str(font_path), font_size)
    if hasattr(font, 'getbbox'):
        w, h = font.getbbox(name)[2:]
    else:
        w, h = font.getsize(name)
    pad = int(sig_cfg.get("padding", 20))
    img = Image.new("RGBA", (w + pad*2, h + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), name, font=font, fill=(0, 0, 0, 255))

    rot_cfg = sig_cfg.get("rotation", {})
    angle = random.uniform(float(rot_cfg.get("min", -10)), float(rot_cfg.get("max", 10)))
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    border_cfg = sig_cfg.get("border", {})
    border = random.randint(int(border_cfg.get("min", 0)), int(border_cfg.get("max", 10)))
    img = ImageOps.expand(img, border=border, fill=(0, 0, 0, 0))

    blur_cfg = sig_cfg.get("blur", {})
    blur = random.uniform(float(blur_cfg.get("min", 0.0)), float(blur_cfg.get("max", 0.8)))
    img = img.filter(ImageFilter.GaussianBlur(blur))
    return img

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic signature PNGs using config")
    parser.add_argument("--config", help="Path to JSON config", required=False)
    parser.add_argument("--num", type=int, help="Override number of signatures", required=False)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
    sig_cfg: Dict[str, Any] = cfg.get("signatures", {})

    # RNG seed (optional)
    seed = sig_cfg.get("seed")
    if seed is not None:
        random.seed(int(seed))
        try:
            Faker.seed(int(seed))
        except Exception:
            pass

    fonts_dir, png_dir = resolve_paths(sig_cfg)
    fonts = choose_fonts(fonts_dir)
    locale = determine_locale(sig_cfg, fonts)

    fake = Faker(locale)

    num_signatures = int(args.num or sig_cfg.get("num", 500))
    print(f"[DEBUG] Generating {num_signatures} signatures to: {png_dir}")
    for i in range(num_signatures):
        name = fake.name()
        font_choice = random.choice(fonts)
        signature_img = render_signature(name, font_choice, sig_cfg)
        out_path = png_dir / f"{uuid.uuid4().hex}.png"
        signature_img.save(out_path)
        if (i + 1) % 50 == 0:
            print(f"[DEBUG] Saved {i + 1}/{num_signatures} signatures")
    total_files = len(list(png_dir.glob("*.png")))
    print(f"[DEBUG] Done. Total signature PNGs in {png_dir}: {total_files}")

if __name__ == '__main__':
    main()
