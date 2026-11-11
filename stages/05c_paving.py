"""Stage 5c: paving textures made in code, matched by eye to street-level photos of the square:
small terracotta pavers round the tower, beige exposed-aggregate concrete on the promenade, grey aggregate in the planter strip."""
import json, os, numpy as np
from PIL import Image, ImageFilter
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = f"{root}/data/assets"; os.makedirs(out, exist_ok=True); N = 1024
rng = np.random.default_rng(3)
def speckle(amp_fine, amp_coarse):
    fine = rng.normal(0, amp_fine, (N, N, 1)); mid = np.asarray(Image.fromarray(rng.integers(0, 255, (N // 3, N // 3), dtype=np.uint8)).resize((N, N), Image.BICUBIC), np.float32)[..., None] / 255 - 0.5
    big = np.asarray(Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).resize((N, N), Image.BICUBIC), np.float32)[..., None] / 255 - 0.5
    return fine + mid * amp_coarse + big * amp_coarse * 0.6
def save(name, a): Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(f"{out}/{name}.jpg", quality=90)
def aggregate(name, base, joint):                     # tile = 4 m, one joint line on two edges
    a = np.ones((N, N, 3)) * np.array(base) / 255 + speckle(0.035, 0.10)
    pebbles = rng.random((N, N)) < 0.012; a[pebbles] *= rng.uniform(0.55, 1.25, (pebbles.sum(), 1))
    a[:4, :] = np.array(joint) / 255; a[:, :4] = np.array(joint) / 255; save(name, a)
def pavers(name, pal, joint, cols=10, rows=20):       # tile = 2 m: 20 x 10 cm pavers in running bond
    y, x = np.mgrid[0:N, 0:N].astype(np.float32); bh = N / rows; bw = N / cols
    r = np.floor(y / bh).astype(int); xo = x + (r % 2) * bw / 2; c = np.floor(xo / bw).astype(int) % cols
    pick = rng.integers(0, len(pal), (rows, cols)); shade = rng.uniform(0.9, 1.08, (rows, cols))
    a = np.array(pal, np.float32)[pick[r, c]] / 255 * shade[r, c][..., None] + speckle(0.03, 0.06)
    fy = y / bh - np.floor(y / bh); fx = xo / bw - np.floor(xo / bw); j = (fy < 2.2 / bh) | (fx < 2.2 / bw)
    a[j] = np.array(joint) / 255; save(name, a)
aggregate("agg_beige", (203, 189, 160), (176, 150, 96)); aggregate("agg_grey", (168, 166, 158), (120, 118, 112))
pavers("pav_terra", [(170, 138, 120), (160, 130, 114), (180, 150, 130), (152, 128, 116), (166, 144, 132), (174, 134, 114), (158, 136, 124)], (118, 104, 94))
man = json.load(open(f"{out}/assets.json"))
for k in ("agg_beige", "agg_grey", "pav_terra"): man[k] = {"diff": f"data/assets/{k}.jpg"}
json.dump(man, open(f"{out}/assets.json", "w"), indent=1); print("paving textures ok")
