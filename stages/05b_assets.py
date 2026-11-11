"""Stage 5b: free CC0 assets from Poly Haven: sky HDRI and PBR textures (cached)."""
import json, os, subprocess
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = f"{root}/data/assets"; os.makedirs(out, exist_ok=True)
TEX = {"stone": "medieval_blocks_05", "brick": "medieval_red_brick", "paving": "rectangular_paving", "prom": "concrete_pavement", "cobble": "cobblestone_floor_04",
       "grass": "grass_ground", "asphalt": "asphalt_02", "plaster": "painted_plaster_wall", "rooftile": "clay_roof_tiles_02", "bark": "pine_bark"}
HDRI = "kloofendal_48d_partly_cloudy_puresky"
def get(url, f):
    if os.path.exists(f) and os.path.getsize(f) > 10000: return
    for _ in range(4):
        r = subprocess.run(["curl", "-sS", "-L", "-m", "300", "-o", f + ".tmp", "-w", "%{http_code}", url], capture_output=True, text=True)
        if r.stdout.strip() == "200": os.replace(f + ".tmp", f); return
    print("FAILED", url, r.stdout, r.stderr)
def api(name): return json.loads(subprocess.run(["curl", "-sS", "-m", "60", f"https://api.polyhaven.com/files/{name}"], capture_output=True, text=True).stdout)
man = {}
for key, name in TEX.items():
    try: f = api(name)
    except Exception as e: print("no api", name, e); continue
    man[key] = {}
    for kind, alts in (("diff", ["Diffuse", "diff"]), ("nor", ["nor_gl"]), ("rough", ["Rough", "rough"])):
        k = next((a for a in alts if a in f), None)
        if not k: continue
        url = f[k]["2k"]["jpg"]["url"]; dst = f"{out}/{key}_{kind}.jpg"; get(url, dst)
        if os.path.exists(dst): man[key][kind] = f"data/assets/{key}_{kind}.jpg"
    print(key, name, list(man[key]))
h = api(HDRI)["hdri"]["4k"]["hdr"]["url"]; get(h, f"{out}/sky.hdr"); man["sky"] = "data/assets/sky.hdr"
json.dump(man, open(f"{out}/assets.json", "w"), indent=1); print("sky MB", round(os.path.getsize(f"{out}/sky.hdr") / 1e6, 1))
