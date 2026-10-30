"""1200x627 LinkedIn featured thumbnail in the portfolio house style (navy header, light canvas)."""
from PIL import Image, ImageDraw, ImageFont
from . import config as C

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"


def make(fig_name="06_var_backtest_market.png", title="Tadawul Sector Risk Study: GARCH & VaR",
         out=C.ROOT / "docs" / "thumbnail.png"):
    W, H, bar = 1200, 627, 90
    img = Image.new("RGB", (W, H), (244, 246, 249))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, bar], fill=(11, 61, 92))
    d.text((30, bar // 2), title, font=ImageFont.truetype(FONT.format("-Bold"), 40), fill="white", anchor="lm")
    fig = Image.open(C.FIG_DIR / fig_name).convert("RGB")
    maxw, maxh = W - 2 * 100, H - bar - 60
    s = min(maxw / fig.width, maxh / fig.height)
    fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
    img.paste(fig, ((W - fig.width) // 2, bar + 12))
    d.text((W - 20, H - 18), "Amir Khan", font=ImageFont.truetype(FONT.format(""), 20), fill=(92, 92, 92), anchor="rs")
    out.parent.mkdir(parents=True, exist_ok=True); img.save(out)
    return out


if __name__ == "__main__":
    print(make())
