"""Stage 5: satellite basemap in three layers (near / city / far). Esri World Imagery. For showing, not for sale."""
import json, os, math, subprocess, sys
from PIL import Image
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(f"{root}/config.json")); a = cfg["anchor"]; b = cfg["bbox"]
R = 6378137.0; kx = math.cos(math.radians(a["lat"])) * math.radians(1) * R; ky = math.radians(1) * R
def tx(lon, z): return (lon + 180) / 360 * 2 ** z
def ty(lat, z): return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z
def lon_of(x, z): return x / 2 ** z * 360 - 180
def lat_of(y, z): return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2 ** z))))
def layer(name, z, w, s, e, n, maxpx=8192):
    X0, X1 = int(tx(w, z)), int(tx(e, z)); Y0, Y1 = int(ty(n, z)), int(ty(s, z))
    d = f"{root}/data/tiles/{z}"; os.makedirs(d, exist_ok=True); jobs = []
    for x in range(X0, X1 + 1):
        for y in range(Y0, Y1 + 1):
            f = f"{d}/{x}_{y}.jpg"
            if not (os.path.exists(f) and os.path.getsize(f) > 500):
                jobs.append(f"url = https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}\noutput = {f}\n")
    for i in range(0, len(jobs), 40):
        subprocess.run(["curl", "-sS", "-m", "120", "--retry", "3", "-Z", "--parallel-max", "8", "-K", "-"], input="".join(jobs[i:i + 40]), text=True, capture_output=True)
    im = Image.new("RGB", ((X1 - X0 + 1) * 256, (Y1 - Y0 + 1) * 256), (60, 80, 100)); missing = 0
    for x in range(X0, X1 + 1):
        for y in range(Y0, Y1 + 1):
            try: im.paste(Image.open(f"{d}/{x}_{y}.jpg").convert("RGB"), ((x - X0) * 256, (y - Y0) * 256))
            except Exception:
                missing += 1
                for up in (1, 2, 3):                  # fall back to the parent tile, cropped and enlarged
                    px, py, zz = x >> up, y >> up, z - up; pf = f"{root}/data/tiles/parent_{zz}_{px}_{py}.jpg"
                    if not os.path.exists(pf): subprocess.run(["curl", "-sS", "-m", "60", "-o", pf, f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zz}/{py}/{px}"])
                    try:
                        k = 256 >> up; ox, oy = (x - (px << up)) * k, (y - (py << up)) * k
                        im.paste(Image.open(pf).convert("RGB").crop((ox, oy, ox + k, oy + k)).resize((256, 256), Image.BICUBIC), ((x - X0) * 256, (y - Y0) * 256)); break
                    except Exception: pass
    if max(im.size) > maxpx: im = im.resize((min(maxpx, im.size[0]), min(maxpx, im.size[1])), Image.LANCZOS)
    im.save(f"{root}/data/sat_{name}.jpg", quality=92)
    ext = [(lon_of(X0, z) - a["lon"]) * kx, (lat_of(Y1 + 1, z) - a["lat"]) * ky, (lon_of(X1 + 1, z) - a["lon"]) * kx, (lat_of(Y0, z) - a["lat"]) * ky]
    print(name, "z", z, "tiles", (X1 - X0 + 1) * (Y1 - Y0 + 1), "missing", missing, "px", im.size, "m/px", round((ext[2] - ext[0]) / im.size[0], 2))
    return {"file": f"data/sat_{name}.jpg", "extent": ext}
r = 520
out = {"near": layer("near", 19, a["lon"] - r / kx, a["lat"] - r / ky, a["lon"] + r / kx, a["lat"] + r / ky),
       "city": layer("city", 17, b["west"], b["south"], b["east"], b["north"]),
       "far": layer("far", 13, a["lon"] - 26000 / kx, a["lat"] - 26000 / ky, a["lon"] + 26000 / kx, a["lat"] + 26000 / ky)}
json.dump(out, open(f"{root}/data/imagery.json", "w"), indent=1)
