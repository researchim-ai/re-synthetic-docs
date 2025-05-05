import random
import uuid
import pathlib
from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

# ------------------------------------------------------------
# Script: make_signatures.py
# Purpose: Generate handwritten-like signature PNGs with alpha channel
# Fix: Fallback to English names if no Cyrillic-supporting font detected
# Usage: python make_signatures.py
# ------------------------------------------------------------

# Resolve paths relative to this script
script_dir = pathlib.Path(__file__).resolve().parent
root = script_dir / "assets" / "signatures"
fonts_dir = root / "fonts"
png_dir = root / "png"
png_dir.mkdir(parents=True, exist_ok=True)

# List all ttf fonts
all_fonts = list(fonts_dir.glob("*.ttf"))
print(f"[DEBUG] looking for fonts in: {fonts_dir}")
print(f"[DEBUG] found fonts: {[f.name for f in all_fonts]}")

# Check Cyrillic support: render a sample Cyrillic string
from PIL import ImageFont

def supports_cyrillic(font_path: pathlib.Path) -> bool:
    try:
        ft = ImageFont.truetype(str(font_path), 64)
        mask = ft.getmask("Тест")
        return mask.getbbox() is not None
    except Exception:
        return False

# Determine locale based on font support
cyrillic_fonts = [f for f in all_fonts if supports_cyrillic(f)]
if cyrillic_fonts:
    print(f"[DEBUG] Cyrillic fonts: {[f.name for f in cyrillic_fonts]}")
    locale = "ru_RU"
else:
    print("[WARN] No Cyrillic-supporting fonts detected. Falling back to English names for signatures.")
    locale = "en_US"

# Use all available fonts for rendering
fonts = all_fonts.copy()
if not fonts:
    print("[ERROR] No .ttf fonts found in signatures/fonts. Please add at least one font.")
    exit(1)

# Initialize Faker with chosen locale
fake = Faker(locale)

# Function to render a signature image for a name
def render_signature(name: str, font_path: pathlib.Path) -> Image.Image:
    font_size = random.randint(48, 72)
    font = ImageFont.truetype(str(font_path), font_size)
    # Measure text size
    if hasattr(font, 'getbbox'):
        w, h = font.getbbox(name)[2:]
    else:
        w, h = font.getsize(name)
    pad = 20
    img = Image.new("RGBA", (w + pad*2, h + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), name, font=font, fill=(0, 0, 0, 255))

    # Apply random rotation, border, and blur
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    border = random.randint(0, 10)
    img = ImageOps.expand(img, border=border, fill=(0, 0, 0, 0))
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.8)))
    return img

# Main generation loop
def main():
    num_signatures = 500
    print(f"[DEBUG] Generating {num_signatures} signatures to: {png_dir}")
    for i in range(num_signatures):
        name = fake.name()
        font_choice = random.choice(fonts)
        signature_img = render_signature(name, font_choice)
        out_path = png_dir / f"{uuid.uuid4().hex}.png"
        signature_img.save(out_path)
        if (i + 1) % 50 == 0:
            print(f"[DEBUG] Saved {i + 1}/{num_signatures} signatures")
    total_files = len(list(png_dir.glob("*.png")))
    print(f"[DEBUG] Done. Total signature PNGs in {png_dir}: {total_files}")

if __name__ == '__main__':
    main()
