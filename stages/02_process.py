"""Stage 2: merge OSM cells, project to local meters, write layers + building heights."""
import json, os, math, random, collections
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString, box, mapping
from shapely.ops import unary_union, polygonize
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(f"{root}/config.json")); a = cfg["anchor"]; b = cfg["bbox"]
R = 6378137.0; kx = math.cos(math.radians(a["lat"])) * math.radians(1) * R; ky = math.radians(1) * R
def xy(lon, lat): return ((lon - a["lon"]) * kx, (lat - a["lat"]) * ky)
nodes, ways, rels = {}, {}, {}
for f in json.load(open(f"{root}/data/osm/files.json")):
    for el in ET.parse(f).getroot():
        i = el.get("id")
        if el.tag == "node":
            nodes[i] = xy(float(el.get("lon")), float(el.get("lat")))
            t = {x.get("k"): x.get("v") for x in el.findall("tag")}
            if t.get("natural") == "tree": ways["n" + i] = ([i], t)
        elif el.tag == "way" and i not in ways:
            ways[i] = ([n.get("ref") for n in el.findall("nd")], {x.get("k"): x.get("v") for x in el.findall("tag")})
        elif el.tag == "relation" and i not in rels:
            rels[i] = ([(m.get("type"), m.get("ref"), m.get("role")) for m in el.findall("member")], {x.get("k"): x.get("v") for x in el.findall("tag")})
x0, y0 = xy(b["west"], b["south"]); x1, y1 = xy(b["east"], b["north"]); clip = box(x0, y0, x1, y1)
def pts(refs): return [nodes[r] for r in refs if r in nodes]
def poly(refs):
    p = pts(refs)
    if len(p) < 4: return None
    g = Polygon(p)
    if not g.is_valid: g = g.buffer(0)
    return g if (not g.is_empty and g.area > 4) else None
def relpoly(members):
    outer = [LineString(pts(ways[r][0])) for t, r, role in members if t == "way" and r in ways and role != "inner" and len(pts(ways[r][0])) > 1]
    inner = [LineString(pts(ways[r][0])) for t, r, role in members if t == "way" and r in ways and role == "inner" and len(pts(ways[r][0])) > 1]
    o = unary_union(list(polygonize(unary_union(outer)))) if outer else None
    if o is None or o.is_empty: return None
    if inner:
        i = unary_union(list(polygonize(unary_union(inner))))
        o = o.difference(i)
    return o
LEVEL = 3.1
DEFAULT = {"apartments": 22, "residential": 20, "yes": 19, "commercial": 20, "retail": 12, "office": 24, "hotel": 26, "church": 14,
           "house": 7, "garage": 3, "garages": 3, "shed": 3, "roof": 4, "kiosk": 3, "school": 12, "university": 16, "hospital": 18,
           "public": 15, "civic": 15, "industrial": 9, "warehouse": 8, "service": 4, "hut": 3, "ruins": 3}
def num(v):
    try: return float(str(v).replace("m", "").replace(",", ".").strip())
    except Exception: return None
stats = collections.Counter(); buildings = []
def add_building(g, t, oid):
    g = g.intersection(clip)
    for part in (g.geoms if hasattr(g, "geoms") else [g]):
        if part.geom_type != "Polygon" or part.area < 6: continue
        h = num(t.get("height")); src = "height"
        if h is None and num(t.get("building:levels")) is not None:
            h = num(t["building:levels"]) * LEVEL + 1.0; src = "levels"
        if h is None:
            rnd = random.Random(int(oid) if oid.isdigit() else hash(oid))
            base = DEFAULT.get(t.get("building", "yes"), 16)
            if part.area < 40: base = min(base, 6)
            h = base * rnd.uniform(0.8, 1.2); src = "estimate"
        stats[src] += 1
        part = part.simplify(0.15)
        buildings.append({"id": oid, "h": round(h, 1), "min_h": num(t.get("min_height")) or 0, "src": src, "kind": t.get("building"),
                          "name": t.get("name:en") or t.get("name"), "hist": t.get("historic"), "poly": mapping(part)["coordinates"]})
L = collections.defaultdict(list)
def area(kind, g, t):
    g = g.intersection(clip)
    if not g.is_empty: L[kind].append({"poly": [mapping(p)["coordinates"] for p in (g.geoms if hasattr(g, "geoms") else [g]) if p.geom_type == "Polygon"], "name": t.get("name")})
GREEN = {"park", "garden", "pitch", "playground"}; GREENLU = {"grass", "forest", "cemetery", "recreation_ground", "village_green", "meadow"}
def classify(g, t, oid):
    if g is None: return
    if "building" in t or "building:part" in t:
        if "building" in t and t.get("building") != "no": add_building(g, t, oid)
    elif t.get("natural") == "water" or t.get("waterway") == "riverbank": area("water", g, t)
    elif t.get("leisure") in GREEN or t.get("landuse") in GREENLU or t.get("natural") in ("wood", "scrub", "grassland"): area("green", g, t)
    elif t.get("highway") == "pedestrian" or t.get("place") == "square": area("plaza", g, t)
coast = []; trees = []
WIDTH = {"motorway": 14, "trunk": 14, "primary": 14, "secondary": 11, "tertiary": 9, "residential": 7, "unclassified": 7, "living_street": 6,
         "service": 4, "pedestrian": 6, "footway": 2.5, "path": 2, "cycleway": 2.5, "steps": 2.5}
for i, (refs, t) in ways.items():
    if i.startswith("n"):
        p = nodes[refs[0]]
        if clip.contains(LineString([p, (p[0] + .1, p[1])])): trees.append([round(p[0], 2), round(p[1], 2)])
        continue
    if t.get("natural") == "coastline": coast.append(pts(refs)); continue
    if "highway" in t and t.get("area") != "yes" and t["highway"] in WIDTH:
        ls = LineString(pts(refs)).intersection(clip) if len(pts(refs)) > 1 else None
        if ls is not None and not ls.is_empty:
            lanes = num(t.get("lanes")); w = max(WIDTH[t["highway"]], (lanes or 0) * 3.3)
            for part in (ls.geoms if hasattr(ls, "geoms") else [ls]):
                if part.geom_type == "LineString": L["roads"].append({"kind": t["highway"], "w": w, "name": t.get("name"), "line": [list(c) for c in part.coords]})
        continue
    if refs and refs[0] == refs[-1]: classify(poly(refs), t, i)
for i, (members, t) in rels.items():
    if t.get("type") == "multipolygon": classify(relpoly(members), t, "r" + i)
# Sea: coastline has land on its left. Split the bbox with the coastline, keep pieces with no buildings.
lines = unary_union([LineString(c) for c in coast if len(c) > 1] + [clip.boundary])
bu = unary_union([Polygon(p["poly"][0]).buffer(0) for p in buildings])
sea = [p for p in polygonize(lines) if clip.contains(p.representative_point()) and p.intersection(bu).area < 0.002 * p.area and p.area > 2000]
# of the building-free pieces keep those touching the south/west edge chain (the gulf)
seaU = unary_union([p for p in sea if p.bounds[1] <= y0 + 1 or p.bounds[0] <= x0 + 1])
L["sea"] = [{"poly": [mapping(p)["coordinates"] for p in (seaU.geoms if hasattr(seaU, "geoms") else [seaU])]}] if not seaU.is_empty else []
out = {"bounds": [x0, y0, x1, y1], "buildings": buildings, "trees": trees, **L}
json.dump(out, open(f"{root}/data/layers.json", "w"))
print("buildings", len(buildings), dict(stats)); print({k: len(v) for k, v in L.items()}, "trees", len(trees), "coast ways", len(coast))
print("sea area km2", round(seaU.area / 1e6, 2), "box km", round((x1 - x0) / 1000, 2), round((y1 - y0) / 1000, 2))
for bd in buildings:
    if bd["name"] and ("Tower" in bd["name"] or "Πύργος" in bd["name"]): print("LANDMARK", bd["id"], bd["name"], bd["h"], bd["src"], bd["kind"])
