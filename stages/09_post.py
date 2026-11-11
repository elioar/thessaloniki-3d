"""Stage 9: finishing in the picture: distance haze from the mist pass, a soft glow on highlights, slight vignette and grain."""
import sys, os, numpy as np
from PIL import Image, ImageFilter
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in sys.argv[1:]:
    base = f"{root}/out/{name}"; im = np.asarray(Image.open(base + "_raw.png").convert("RGB")).astype(np.float32) / 255; mist = np.load(base + "_mist.npy")
    if mist.shape[:2] != im.shape[:2]: mist = np.asarray(Image.fromarray(mist).resize((im.shape[1], im.shape[0])))
    if abs(float(mist[0].mean()) - 1) > abs(float(mist[-1].mean()) - 1): mist = mist[::-1]          # sky (mist = 1) belongs at the top
    d = mist * 30000.0; sky = np.asarray(Image.fromarray((mist > 0.999).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0   # keep haze off the silhouettes
    f = (1 - np.exp(-d / 9000.0)) * 0.85; f[sky] = 0; haze = np.array([0.74, 0.80, 0.86], np.float32)
    out = im * (1 - f[..., None]) + haze * f[..., None]
    lum = out.max(-1); hi = np.clip((lum - 0.82) / 0.18, 0, 1)[..., None] * out
    glow = np.asarray(Image.fromarray((hi * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(im.shape[1] / 110))).astype(np.float32) / 255
    out = 1 - (1 - out) * (1 - glow * 0.35)
    h, w = lum.shape; yy, xx = np.mgrid[0:h, 0:w]; r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    out *= (1 - 0.10 * r2)[..., None]; out += np.random.default_rng(1).normal(0, 0.006, out.shape)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(base + ".png"); print("post", name)
