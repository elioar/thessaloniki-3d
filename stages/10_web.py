"""Stage 10: export for the browser. Plain typed arrays + a manifest (docs/data), small textures (docs/tex).
Blender is Z-up, the browser is Y-up: (x, y, z) -> (x, z, -y)."""
import json, os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = f"{root}/docs"; os.makedirs(f"{out}/data", exist_ok=True); os.makedirs(f"{out}/tex", exist_ok=True)
site = json.load(open(f"{root}/data/tower_site.json")); T = np.array([site["x"], site["y"], site["ground"] - 0.15], np.float32)
man = {"meshes": [], "tower": [float(T[0]), float(T[2]), float(-T[1])]}
def export(name, src, mats, offset=None, drop=()):
    """one file per material group: positions f32, [uv f32], [colour u8 x4], indices u32."""
    d = np.load(f"{root}/data/geo_{src}.npz"); v = d["v"].copy(); t = d["t"]; m = d["mat"]
    if offset is not None: v += offset
    for mi, (mname, want_uv, want_col) in mats.items():
        if mi in drop: continue
        tt = t[m == mi]
        if not len(tt): continue
        used, inv = np.unique(tt, return_inverse=True); idx = inv.reshape(-1, 3).astype(np.uint32); p = v[used]
        pos = np.c_[p[:, 0], p[:, 2], -p[:, 1]].astype(np.float32); parts = [pos.tobytes()]; rec = {"name": f"{name}_{mname}", "material": mname, "verts": len(used), "tris": len(idx)}
        if want_uv: parts.append(d["uv"][used].astype(np.float32).tobytes()); rec["uv"] = True
        if want_col: parts.append(np.c_[np.clip(d["col"][used], 0, 1) * 255, np.full(len(used), 255)].astype(np.uint8).tobytes()); rec["color"] = True
        parts.append(idx.tobytes()); f = f"data/{name}_{mname}.bin"; open(f"{out}/{f}", "wb").write(b"".join(parts)); rec["file"] = f; man["meshes"].append(rec)
        print(rec["name"], rec["verts"], rec["tris"], round(os.path.getsize(f"{out}/{f}") / 1e6, 2), "MB")
export("ground", "ground", {0: ("ground", False, False)})
export("quay", "quay", {0: ("quay", False, False)})
export("bld", "buildings", {0: ("wall", True, True), 1: ("roof", False, True), 2: ("plain", False, True), 3: ("glass", False, True)})
export("tower", "tower", {0: ("stone", True, False), 1: ("dark", False, False), 2: ("pave", False, False), 3: ("white", False, False), 4: ("blue", False, False), 5: ("metal", False, False), 6: ("brick", True, False), 7: ("cap", False, False)}, offset=T)
export("site", "site", {0: ("grass", True, False), 1: ("paving", True, False), 2: ("cobble", True, False), 3: ("asphalt", True, False), 4: ("kerb", False, False), 5: ("prom", True, False),
                        6: ("lamp", False, False), 7: ("glow", False, False), 8: ("car", False, True), 9: ("carglass", False, False), 10: ("bark", False, False)}, drop=(11,))
W = json.load(open(f"{root}/data/web_trees.json"))
np.array(W["trees"], np.float32).tofile(f"{out}/data/trees.bin"); np.array(W["clumps"], np.float32).tofile(f"{out}/data/clumps.bin")
man["trees"] = {"file": "data/trees.bin", "count": len(W["trees"]), "stride": 8}; man["clumps"] = {"file": "data/clumps.bin", "count": len(W["clumps"]), "stride": 8}
img = json.load(open(f"{root}/data/imagery.json")); man["sat"] = {}
for key, size in (("near", 4096), ("city", 2048), ("far", 2048)):
    im = Image.open(f"{root}/{img[key]['file']}").convert("RGB").resize((size, size), Image.LANCZOS); im.save(f"{out}/tex/sat_{key}.jpg", quality=84); man["sat"][key] = {"file": f"tex/sat_{key}.jpg", "extent": img[key]["extent"]}
A = json.load(open(f"{root}/data/assets/assets.json"))
for key in ("stone", "brick", "prom", "cobble", "grass", "asphalt"):
    for kind in ("diff", "nor"):
        Image.open(f"{root}/{A[key][kind]}").convert("RGB").resize((1024, 1024), Image.LANCZOS).save(f"{out}/tex/{key}_{kind}.jpg", quality=82)
json.dump(man, open(f"{out}/data/manifest.json", "w"), indent=1)
tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs); print("docs total MB", round(tot / 1e6, 1))
