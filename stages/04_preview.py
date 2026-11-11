"""Stage 4: check map. Buildings by height, sea, green, roads."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = json.load(open(f"{root}/data/layers.json")); cfg = json.load(open(f"{root}/config.json"))
fig, ax = plt.subplots(figsize=(12, 12), dpi=130); x0, y0, x1, y1 = L["bounds"]
ax.set_facecolor("#efece6")
def polys(items): return [ring[0] for it in items for ring in it["poly"]]
ax.add_collection(PolyCollection(polys(L["sea"]), facecolor="#a9c9de", edgecolor="none"))
ax.add_collection(PolyCollection(polys(L["green"]), facecolor="#bcd9a8", edgecolor="none"))
ax.add_collection(PolyCollection(polys(L.get("plaza", [])), facecolor="#ddd6c8", edgecolor="none"))
ax.add_collection(LineCollection([r["line"] for r in L["roads"]], colors="#ffffff", linewidths=[r["w"] / 6 for r in L["roads"]]))
pc = PolyCollection([bd["poly"][0] for bd in L["buildings"]], array=[bd["h"] for bd in L["buildings"]], cmap="viridis", edgecolor="none"); pc.set_clim(0, 35)
ax.add_collection(pc); plt.colorbar(pc, ax=ax, shrink=.6, label="height (m)")
ax.plot(0, 0, "r*", ms=18); ax.add_patch(plt.Circle((0, 0), cfg["inner_radius_m"], fill=False, ec="red", lw=1.2, ls="--"))
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_title(f"Thessaloniki check map: {len(L['buildings'])} buildings (star = White Tower, dashed = detail zone)")
fig.savefig(f"{root}/out/01_check_map.png", bbox_inches="tight"); print("ok")
