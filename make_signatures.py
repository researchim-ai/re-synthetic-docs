import random, pathlib, uuid
from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

root = pathlib.Path("assets/signatures")
fonts = list((root / "fonts").glob("*.ttf"))
out_dir = root / "png"
out_dir.mkdir(parents=True, exist_ok=True)

fake = Faker("ru_RU")

def render_signature(name: str, font_path: pathlib.Path):
    font_size = random.randint(48, 72)
    font = ImageFont.truetype(str(font_path), font_size)
    w, h = font.getbbox(name)[2:]
    pad = 20
    img = Image.new("RGBA", (w + pad*2, h + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), name, font=font, fill=(0, 0, 0, 255))

    # лёгкая перспектива / кривизна
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, Image.BICUBIC, expand=True)
    img = ImageOps.expand(img, border=random.randint(0, 10), fill=(0,0,0,0))
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.8)))
    return img

for _ in range(500):          # сколько подписей сделать
    name = fake.name()
    img = render_signature(name, random.choice(fonts))
    img.save(out_dir / f"{uuid.uuid4().hex}.png")
