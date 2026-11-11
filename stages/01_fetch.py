"""Stage 1: download OSM data for the bbox in small cells from the OSM API (cached)."""
import json, os, subprocess, sys, time
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(root, "config.json")))
b = cfg["bbox"]; out = os.path.join(root, "data", "osm"); os.makedirs(out, exist_ok=True)
STEP = 0.006
def frange(a, z):
    v = a
    while v < z - 1e-9:
        yield round(v, 6), round(min(v + STEP, z), 6); v += STEP
cells = [(w, s, e, n) for w, e in frange(b["west"], b["east"]) for s, n in frange(b["south"], b["north"])]
def fetch(w, s, e, n, depth=0):
    f = os.path.join(out, f"{w}_{s}_{e}_{n}.osm")
    if os.path.exists(f) and os.path.getsize(f) > 200: return [f]
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
    for attempt in range(4):
        r = subprocess.run(["curl", "-sS", "-m", "120", "-o", f + ".tmp", "-w", "%{http_code}", url], capture_output=True, text=True)
        if r.stdout.strip() == "200":
            os.replace(f + ".tmp", f); time.sleep(1); return [f]
        if r.stdout.strip() == "400" and depth < 3:   # too many nodes: split in four
            mx, my = round((w + e) / 2, 6), round((s + n) / 2, 6)
            return sum((fetch(*c, depth + 1) for c in [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]), [])
        time.sleep(2 ** (attempt + 1))
    sys.exit(f"failed {url}: {r.stdout} {r.stderr}")
files = []
for i, c in enumerate(cells):
    files += fetch(*c); print(f"{i+1}/{len(cells)}", flush=True)
json.dump(files, open(os.path.join(out, "files.json"), "w"))
print("cells", len(cells), "files", len(files), "MB", round(sum(os.path.getsize(f) for f in files) / 1e6, 1))
