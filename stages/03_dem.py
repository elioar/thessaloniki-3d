"""Stage 3: SRTM 30 m elevation on a coarse grid via OpenTopoData, smoothed to bare-earth-ish."""
import json, os, subprocess, time, math
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(f"{root}/config.json")); b = cfg["bbox"]; N = 24
out = f"{root}/data/dem.json"
lons = np.linspace(b["west"], b["east"], N); lats = np.linspace(b["south"], b["north"], N)
pts = [(la, lo) for la in lats for lo in lons]; z = []
for i in range(0, len(pts), 100):
    q = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in pts[i:i + 100])
    for attempt in range(4):
        r = subprocess.run(["curl", "-sS", "-m", "60", f"https://api.opentopodata.org/v1/srtm30m?locations={q}"], capture_output=True, text=True)
        try:
            z += [p["elevation"] or 0 for p in json.loads(r.stdout)["results"]]; break
        except Exception: time.sleep(3)
    else: raise SystemExit("dem failed " + r.stdout[:200])
    time.sleep(1.2)
Z = np.array(z, float).reshape(N, N)
# bare-earth-ish: local minimum then blur (SRTM sees rooftops)
P = np.pad(Z, 1, mode="edge"); Zm = np.min([P[i:i + N, j:j + N] for i in range(3) for j in range(3)], axis=0)
for _ in range(3):
    P = np.pad(Zm, 1, mode="edge"); Zm = sum(P[i:i + N, j:j + N] for i in range(3) for j in range(3)) / 9
Zm = np.maximum(Zm, 0)
json.dump({"n": N, "z": Zm.round(2).tolist()}, open(out, "w"))
print("raw min/max", Z.min(), Z.max(), "smoothed", Zm.min().round(1), Zm.max().round(1))
