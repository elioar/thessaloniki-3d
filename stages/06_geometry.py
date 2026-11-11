"""Stage 6: plain-array geometry (ground, quay, buildings, trees). Blender only loads these."""
import json, os, math, random, subprocess, time
import numpy as np, mapbox_earcut as earcut
from shapely.geometry import Polygon, Point, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(f"{root}/config.json")); a = cfg["anchor"]
L = json.load(open(f"{root}/data/layers.json")); x0, y0, x1, y1 = L["bounds"]
R = 6378137.0; kx = math.cos(math.radians(a["lat"])) * math.radians(1) * R; ky = math.radians(1) * R
QUAY = 1.6; FAR = 26000.0; TOWER_ID = "730026417"
# ---------- elevation ----------
D = json.load(open(f"{root}/data/dem.json")); Zi = np.array(D["z"]); N = D["n"]
farf = f"{root}/data/dem_far.json"
if not os.path.exists(farf):
    M = 24; pts = [(a["lat"] + y / ky, a["lon"] + x / kx) for y in np.linspace(-FAR, FAR, M) for x in np.linspace(-FAR, FAR, M)]; z = []
    for i in range(0, len(pts), 100):
        q = "|".join(f"{la:.5f},{lo:.5f}" for la, lo in pts[i:i + 100])
        for _ in range(4):
            r = subprocess.run(["curl", "-sS", "-m", "60", f"https://api.opentopodata.org/v1/srtm30m?locations={q}"], capture_output=True, text=True)
            try: z += [p["elevation"] or 0 for p in json.loads(r.stdout)["results"]]; break
            except Exception: time.sleep(3)
        else: raise SystemExit("far dem failed")
        time.sleep(1.2)
    json.dump({"n": M, "z": np.array(z, float).reshape(M, M).tolist()}, open(farf, "w"))
Zf = np.array(json.load(open(farf))["z"]); Mf = Zf.shape[0]
def bil(Z, u, v):
    n = Z.shape[0]; u = min(max(u, 0), 1) * (n - 1); v = min(max(v, 0), 1) * (n - 1)
    i = min(int(u), n - 2); j = min(int(v), n - 2); fu = u - i; fv = v - j
    return (Z[j, i] * (1 - fu) + Z[j, i + 1] * fu) * (1 - fv) + (Z[j + 1, i] * (1 - fu) + Z[j + 1, i + 1] * fu) * fv
TW = next(Polygon(bd["poly"][0]).centroid for bd in L["buildings"] if bd["id"] == TOWER_ID); TX, TY = TW.x, TW.y; FLAT = 2.4
def land_h(x, y):
    h = max(bil(Zi, (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)), 0.0) + QUAY
    w = min(max((math.hypot(x - TX, y - TY) - 270) / 150, 0), 1); w = w * w * (3 - 2 * w)   # the real plaza is level
    return FLAT * (1 - w) + h * w
def far_h(x, y):
    d = bil(Zf, (x + FAR) / (2 * FAR), (y + FAR) / (2 * FAR)); return d + QUAY if d > 0.5 else 0.0
clip = box(x0, y0, x1, y1)
sea = unary_union([Polygon(p[0], p[1:]) for it in L["sea"] for p in it["poly"]]).buffer(0); psea = prep(sea)
def tri(poly):
    rings = [np.array(poly.exterior.coords[:-1])] + [np.array(r.coords[:-1]) for r in poly.interiors]
    v = np.vstack(rings); idx = earcut.triangulate_float64(v, np.cumsum([len(r) for r in rings]).astype(np.uint32))
    return v, idx.reshape(-1, 3)
class Mesh:
    def __init__(s): s.v = []; s.t = []; s.uv = []; s.col = []; s.mat = []; s.n = 0
    def add(s, verts, tris, uv=None, col=(1, 1, 1), mat=0):
        verts = np.asarray(verts, float); tris = np.asarray(tris, np.int64)
        s.v.append(verts); s.t.append(tris + s.n); s.n += len(verts)
        s.uv.append(np.asarray(uv, float) if uv is not None else verts[:, :2].copy())
        s.col.append(np.tile(np.asarray(col, float), (len(verts), 1))); s.mat.append(np.full(len(tris), mat, np.int32))
    def save(s, name):
        np.savez(f"{root}/data/geo_{name}.npz", v=np.vstack(s.v).astype(np.float32), t=np.vstack(s.t).astype(np.int32),
                 uv=np.vstack(s.uv).astype(np.float32), col=np.vstack(s.col).astype(np.float32), mat=np.concatenate(s.mat))
        print(name, "verts", s.n, "tris", sum(len(t) for t in s.t))
# ---------- ground ----------
g = Mesh(); xs_in = np.linspace(x0, x1, 153); ys_in = np.linspace(y0, y1, 152)
far_lines = np.concatenate([-np.geomspace(FAR, 300, 26)[:-1], np.geomspace(300, FAR, 26)[1:]])
xs = np.array(sorted(set(xs_in.tolist()) | {v + (x0 if v < 0 else x1) for v in far_lines}))
ys = np.array(sorted(set(ys_in.tolist()) | {v + (y0 if v < 0 else y1) for v in far_lines}))
Q = np.array([[0, 1, 2], [0, 2, 3]])
for i in range(len(xs) - 1):
    for j in range(len(ys) - 1):
        xa, xb, ya, yb = xs[i], xs[i + 1], ys[j], ys[j + 1]; c = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
        inside = xa >= x0 - 1e-6 and xb <= x1 + 1e-6 and ya >= y0 - 1e-6 and yb <= y1 + 1e-6
        if not inside:
            def h(p): return land_h(*p) if (x0 - 1e-6 <= p[0] <= x1 + 1e-6 and y0 - 1e-6 <= p[1] <= y1 + 1e-6 and not psea.contains(Point(p).buffer(0.01))) else (0.0 if (x0 - 1e-6 <= p[0] <= x1 + 1e-6 and y0 - 1e-6 <= p[1] <= y1 + 1e-6) else far_h(*p))
            g.add([(p[0], p[1], h(p)) for p in c], Q); continue
        cell = box(xa, ya, xb, yb)
        if psea.contains(cell): g.add([(p[0], p[1], 0.0) for p in c], Q)
        elif not psea.intersects(cell): g.add([(p[0], p[1], land_h(*p)) for p in c], Q)
        else:
            for part, isl in ((cell.difference(sea), True), (cell.intersection(sea), False)):
                for pp in (part.geoms if hasattr(part, "geoms") else [part]):
                    if pp.geom_type == "Polygon" and pp.area > 1e-4:
                        v, t = tri(pp); g.add([(p[0], p[1], land_h(*p) if isl else 0.0) for p in v], t)
g.save("ground")
# ---------- quay wall ----------
q = Mesh(); coast = sea.boundary.difference(clip.boundary.buffer(0.5))
for ls in (coast.geoms if hasattr(coast, "geoms") else [coast]):
    c = list(ls.coords); u = 0
    for p, r in zip(c[:-1], c[1:]):
        d = math.dist(p, r); hp, hr = land_h(*p), land_h(*r)
        q.add([(p[0], p[1], -2), (r[0], r[1], -2), (r[0], r[1], hr), (p[0], p[1], hp)], Q, uv=[(u, 0), (u + d, 0), (u + d, hr + 2), (u, hp + 2)]); u += d
q.save("quay")
def obox(mesh, c, u, length, width, z0, z1, col, mat):
    """box centred on c, lying along the unit vector u."""
    ux, uy = u; nx, ny = -uy, ux; hl, hw = length / 2, width / 2
    b = [(c[0] + sx * hl * ux + sy * hw * nx, c[1] + sx * hl * uy + sy * hw * ny) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    v = [(x, y, z0) for x, y in b] + [(x, y, z1) for x, y in b]
    t = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    mesh.add(v, t, col=col, mat=mat)
# ---------- buildings ----------
PAL = [(0.80, 0.76, 0.68), (0.72, 0.68, 0.60), (0.84, 0.81, 0.75), (0.70, 0.63, 0.52), (0.78, 0.70, 0.56), (0.62, 0.61, 0.58), (0.82, 0.72, 0.60), (0.72, 0.56, 0.44), (0.86, 0.84, 0.80), (0.76, 0.66, 0.50), (0.66, 0.60, 0.54), (0.80, 0.62, 0.50)]
NEARPAL = [(0.74, 0.70, 0.62), (0.72, 0.62, 0.44), (0.68, 0.54, 0.36), (0.62, 0.60, 0.56), (0.78, 0.68, 0.52), (0.70, 0.48, 0.34), (0.60, 0.55, 0.44), (0.80, 0.77, 0.72), (0.66, 0.58, 0.40), (0.54, 0.56, 0.58)]
ROOF = [(0.55, 0.54, 0.52), (0.62, 0.60, 0.57), (0.48, 0.47, 0.46), (0.66, 0.63, 0.58)]
FABRIC = [(0.08, 0.26, 0.14), (0.70, 0.58, 0.38), (0.70, 0.30, 0.08), (0.10, 0.18, 0.40), (0.50, 0.09, 0.08), (0.80, 0.77, 0.68), (0.20, 0.36, 0.30)]
WALLM, ROOFM, PLAIN, GLASS = 0, 1, 2, 3
THEATRE_KTHBE, THEATRE_ROYAL, OFFICERS = "19790720", "19855083", "182044285"
bm = Mesh(); polys = []; tower = None; INNER = 420.0
allp = [Polygon(bd["poly"][0], bd["poly"][1:]).buffer(0) for bd in L["buildings"]]; alltree = STRtree(allp)
def walls(P, zb, zt, zg, col, mat, rnd):
    for ring in [P.exterior] + list(P.interiors):
        c = list(ring.coords); u = rnd.uniform(0, 3)
        for p, r in zip(c[:-1], c[1:]):
            d = math.dist(p, r)
            bm.add([(p[0], p[1], zb), (r[0], r[1], zb), (r[0], r[1], zt), (p[0], p[1], zt)], Q, uv=[(u, zb - zg), (u + d, zb - zg), (u + d, zt - zg), (u, zt - zg)], col=col, mat=mat); u += d
def roof(P, z, col, mat=ROOFM):
    v, t = tri(P); bm.add(np.c_[v, np.full(len(v), z)], t, col=col, mat=mat)
def parapet(P, z, col, h=0.9):
    c = list(P.exterior.coords)
    for p, r in zip(c[:-1], c[1:]): bm.add([(p[0], p[1], z), (r[0], r[1], z), (r[0], r[1], z + h), (p[0], p[1], z + h)], Q, col=col, mat=PLAIN)
def mrr_sides(P):
    """sides of the smallest enclosing rectangle: (mid, along, outward normal, length), pushed out to the real footprint."""
    c = list(P.minimum_rotated_rectangle.exterior.coords)[:4]; cen = P.centroid; out = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]; d = math.dist(a, b); ux, uy = (b[0] - a[0]) / d, (b[1] - a[1]) / d; nx, ny = uy, -ux
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if (mx - cen.x) * nx + (my - cen.y) * ny < 0: nx, ny = -nx, -ny
        out.append(((mx, my), (ux, uy), (nx, ny), d))
    return out
def polykatoikia(P, bd, zg, zt, wall, rnd):
    """Greek apartment block: balconies on the street sides, awnings, set-back top floor, roof clutter."""
    body = P; top = zt
    if bd["h"] > 17 and rnd.random() < 0.65:
        inner = P.buffer(-rnd.uniform(1.8, 2.6), join_style=2)
        if inner.geom_type == "Polygon" and inner.area > 0.45 * P.area:
            top = zt - 3.1; walls(inner, top, zt, zg, wall, WALLM, rnd); roof(inner, zt, ROOF[rnd.randrange(4)]); parapet(inner, zt, wall, 0.5); body = inner
    walls(P, zg - 1.5, top, zg, wall, WALLM, rnd); roof(P, top, ROOF[rnd.randrange(4)]); parapet(P, top, wall, 0.95 if body is not P else 0.8)
    ext = P.exterior if P.exterior.is_ccw else P.exterior.reverse(); c = list(ext.coords); nfl = int((top - zg - 3.6) / 3.1 + 0.3)
    white = tuple(min(1, k * 1.04 + 0.02) for k in wall); deep = rnd.uniform(1.2, 1.7); style = rnd.choice(["solid", "solid", "rail", "segment", "segment"])
    fab = FABRIC[rnd.randrange(len(FABRIC))]; pawn = rnd.choice([0.15, 0.3, 0.45, 0.6])
    for p, r in zip(c[:-1], c[1:]):
        d = math.dist(p, r)
        if d < 4.5: continue
        ux, uy = (r[0] - p[0]) / d, (r[1] - p[1]) / d; nx, ny = uy, -ux; mid = Point((p[0] + r[0]) / 2 + nx * 2.0, (p[1] + r[1]) / 2 + ny * 2.0)
        if any(allp[k].contains(mid) for k in alltree.query(mid)): continue                       # party wall, no balconies
        nb = max(1, int((d - 0.8) / 3.4)); bw = (d - 0.8) / nb
        for k in range(1, nfl + 1):
            z = zg + 3.6 + (k - 1) * 3.1 + 0.02
            spans = [(0.4, d - 0.4)] if style != "segment" else [(0.4 + i * bw + 0.35, 0.4 + (i + 1) * bw - 0.35) for i in range(nb)]
            for s0, s1 in spans:
                m = (s0 + s1) / 2; ln = s1 - s0; cx, cy = p[0] + ux * m, p[1] + uy * m
                obox(bm, (cx + nx * deep / 2, cy + ny * deep / 2), (ux, uy), ln, deep, z - 0.16, z, white, PLAIN)
                if style == "rail":
                    obox(bm, (cx + nx * (deep - 0.04), cy + ny * (deep - 0.04)), (ux, uy), ln, 0.05, z + 0.95, z + 1.02, (0.12, 0.12, 0.12), PLAIN)
                    obox(bm, (cx + nx * (deep - 0.04), cy + ny * (deep - 0.04)), (ux, uy), ln, 0.03, z + 0.1, z + 0.95, (0.30, 0.36, 0.38), GLASS)
                else:
                    obox(bm, (cx + nx * (deep - 0.05), cy + ny * (deep - 0.05)), (ux, uy), ln, 0.1, z, z + 0.95, white, PLAIN)
                    for e in (s0, s1): obox(bm, (p[0] + ux * e + nx * deep / 2, p[1] + uy * e + ny * deep / 2), (ux, uy), 0.1, deep, z, z + 0.95, white, PLAIN)
            for i in range(nb):                                                                     # awnings over some bays
                if rnd.random() > pawn: continue
                f = fab if rnd.random() < 0.8 else FABRIC[rnd.randrange(len(FABRIC))]; a0, a1 = 0.4 + i * bw + 0.25, 0.4 + (i + 1) * bw - 0.25
                A = [(p[0] + ux * a0, p[1] + uy * a0, z + 2.75), (p[0] + ux * a1, p[1] + uy * a1, z + 2.75),
                     (p[0] + ux * a1 + nx * (deep + 0.15), p[1] + uy * a1 + ny * (deep + 0.15), z + 2.05), (p[0] + ux * a0 + nx * (deep + 0.15), p[1] + uy * a0 + ny * (deep + 0.15), z + 2.05)]
                bm.add(A + [(A[2][0], A[2][1], z + 1.85), (A[3][0], A[3][1], z + 1.85)], [(0, 1, 2), (0, 2, 3), (3, 2, 4), (3, 4, 5)], col=f, mat=PLAIN)
    rp = body.buffer(-1.6)                                                                          # roof: stair bulkhead + solar heaters
    if rp.geom_type == "Polygon" and rp.area > 20:
        q = rp.representative_point(); obox(bm, (q.x, q.y), (1, 0), 4.5, 3.5, zt, zt + 2.7, wall, PLAIN); bx = rp.bounds; a = rnd.uniform(0, 3.14)
        for _ in range(min(9, int(rp.area / 45))):
            q = Point(rnd.uniform(bx[0], bx[2]), rnd.uniform(bx[1], bx[3]))
            if rp.contains(q):
                obox(bm, (q.x, q.y), (math.cos(a), math.sin(a)), 2.0, 1.1, zt + 0.35, zt + 0.5, (0.03, 0.04, 0.07), GLASS)
                obox(bm, (q.x - math.sin(a) * 0.8, q.y + math.cos(a) * 0.8), (math.cos(a), math.sin(a)), 1.4, 0.45, zt + 0.7, zt + 1.15, (0.85, 0.85, 0.85), PLAIN)
def theatre_kthbe(P, bd, zg, rnd):
    """Society for Macedonian Studies / State Theatre: pale stone block with a tall pillared front facing the tower."""
    stone = (0.76, 0.70, 0.58); zt = zg + 21.5
    walls(P, zg - 1.5, zt, zg, stone, WALLM, rnd); roof(P, zt, (0.6, 0.58, 0.55)); parapet(P, zt, stone, 1.0)
    inner = P.buffer(-4.5, join_style=2)
    if inner.geom_type == "Polygon": walls(inner, zt, zt + 3.4, zg, stone, WALLM, rnd); roof(inner, zt + 3.4, (0.58, 0.57, 0.55)); parapet(inner, zt + 3.4, stone, 0.5)
    (mid, u, n, d) = max(mrr_sides(P), key=lambda sd: (TX - sd[0][0]) * sd[2][0] + (TY - sd[0][1]) * sd[2][1])
    off = max((x - mid[0]) * n[0] + (y - mid[1]) * n[1] for x, y in P.exterior.coords)                # push to the real wall
    def at(a, o): return (mid[0] + u[0] * a + n[0] * (off + o), mid[1] + u[1] * a + n[1] * (off + o))
    W = d * 0.66; z0, z1 = zg + 6.2, zg + 18.2
    obox(bm, at(0, 0.9), u, W + 3, 1.8, zg, z0, stone, PLAIN)                                         # base with the entrances
    for i in range(-3, 4): obox(bm, at(i * W / 8, 1.82), u, W / 11, 0.08, zg + 0.2, zg + 3.6, (0.03, 0.035, 0.04), GLASS)
    obox(bm, at(0, 1.84), u, W * 0.9, 0.06, zg + 4.0, zg + 5.6, (0.30, 0.42, 0.12), PLAIN)            # the long banner
    obox(bm, at(0, 0.12), u, W, 0.24, z0, z1, (0.05, 0.06, 0.08), GLASS)                              # glazed wall behind the pillars
    npil = 10
    for i in range(npil): obox(bm, at(-W / 2 + W * (i + 0.5) / npil - W / (2 * npil) * 0 , 1.25), u, 0.8, 0.8, z0, z1, stone, PLAIN)
    for sgn in (-1, 1): obox(bm, at(sgn * (W / 2 + 0.9), 0.9), u, 1.8, 1.8, z0, z1, stone, PLAIN)      # end piers
    obox(bm, at(0, 0.9), u, W + 3.6, 2.2, z1, z1 + 1.9, stone, PLAIN); obox(bm, at(0, 1.0), u, W + 4.2, 2.6, z1 + 1.9, z1 + 2.2, (0.9, 0.86, 0.76), PLAIN)
    for i in range(6): obox(bm, at(0, 2.2 + i * 0.35), u, W + 3, 0.35, zg - 0.2, zg + 0.9 - i * 0.16, (0.7, 0.67, 0.6), PLAIN)   # steps
def theatre_royal(P, bd, zg, rnd):
    """Royal Theatre: long low hall in the park, shallow metal roof, a row of piers along each long side."""
    conc = (0.66, 0.64, 0.59); zt = zg + 9.5; walls(P, zg - 1.5, zt, zg, conc, WALLM, rnd)
    sides = mrr_sides(P); long = max(sides, key=lambda sd: sd[3]); short = min(sides, key=lambda sd: sd[3]); cen = P.centroid
    u = long[1]; n = long[2]; hl = long[3] / 2 + 1.5; hw = short[3] / 2 + 4.0; metal = (0.50, 0.52, 0.54)
    def pt(a, b, z): return (cen.x + u[0] * a + n[0] * b, cen.y + u[1] * a + n[1] * b, z)
    v = [pt(-hl, -hw, zt), pt(hl, -hw, zt), pt(hl, 0, zt + 3.2), pt(-hl, 0, zt + 3.2), pt(hl, hw, zt), pt(-hl, hw, zt)]
    bm.add(v, [(0, 1, 2), (0, 2, 3), (3, 2, 4), (3, 4, 5), (1, 4, 2), (0, 3, 5)], col=metal, mat=ROOFM)
    bm.add([pt(-hl, -hw, zt - 0.5), pt(hl, -hw, zt - 0.5), pt(hl, hw, zt - 0.5), pt(-hl, hw, zt - 0.5)], [(0, 2, 1), (0, 3, 2)], col=conc, mat=PLAIN)
    for sgn in (-1, 1):
        obox(bm, pt(0, sgn * (hw - 0.3), 0)[:2], u, 2 * hl, 0.5, zt - 0.5, zt + 0.1, conc, PLAIN)
        for i in range(int(2 * hl / 5.5) + 1):
            a = -hl + 0.8 + i * (2 * hl - 1.6) / int(2 * hl / 5.5); obox(bm, pt(a, sgn * (hw - 0.6), 0)[:2], u, 0.7, 0.7, zg, zt - 0.5, conc, PLAIN)
for bd, P in zip(L["buildings"], allp):
    if bd["id"] == TOWER_ID:
        c = P.centroid; tower = {"x": c.x, "y": c.y, "r_osm": math.sqrt(P.area / math.pi), "ground": land_h(c.x, c.y)}; continue
    if P.is_empty or P.geom_type != "Polygon": continue
    polys.append(P); rnd = random.Random(bd["id"]); dist = P.centroid.distance(TW)
    if bd["src"] == "estimate" and P.area < 110 and dist < 300: bd["h"] = 3.2 if P.area < 70 else 6.0   # kiosks and sheds, not flats
    near = dist < INNER
    wall = (NEARPAL if near else PAL)[rnd.randrange(len(NEARPAL if near else PAL))]; k = rnd.uniform(0.9, 1.05); wall = tuple(min(1, c * k) for c in wall)
    small = P.area < 45 or bd["h"] < 5
    if bd["kind"] in ("church", "cathedral", "chapel") or bd.get("hist"): wall = (0.72, 0.56, 0.44)
    hs = [land_h(*p) for p in P.exterior.coords]; zb = min(hs) - 1.5; zg = sum(hs) / len(hs); zt = zg + bd["h"]
    if bd["id"] == THEATRE_KTHBE: theatre_kthbe(P, bd, zg, rnd); continue
    if bd["id"] == THEATRE_ROYAL: theatre_royal(P, bd, zg, rnd); continue
    if bd["id"] == OFFICERS:                                                    # tiled hipped roof
        wall = (0.86, 0.80, 0.66); walls(P, zb, zt, zg, wall, WALLM, rnd); c = list(P.exterior.coords); q = P.centroid
        for p, r in zip(c[:-1], c[1:]): bm.add([(p[0], p[1], zt), (r[0], r[1], zt), (q.x, q.y, zt + 4.5)], [(0, 1, 2)], col=(0.52, 0.20, 0.10), mat=PLAIN)
        continue
    if near and bd["h"] > 11 and not small and bd["kind"] not in ("church", "school", "sports_centre", "theatre", "military") and not bd["min_h"]:
        polykatoikia(P, bd, zg, zt, wall, rnd); continue
    if bd["min_h"]: zb = zg + bd["min_h"]
    walls(P, zb, zt, zg, wall, PLAIN if small else WALLM, rnd); roof(P, zt, ROOF[rnd.randrange(len(ROOF))])
bm.save("buildings")
json.dump(tower, open(f"{root}/data/tower_site.json", "w")); print("tower site", tower)
# ---------- trees: placed where the satellite photo shows canopy ----------
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
IMG = json.load(open(f"{root}/data/imagery.json")); rnd = random.Random(7); btree = STRtree(polys); spots = []
def canopy(key):
    im = np.asarray(Image.open(f"{root}/{IMG[key]['file']}").convert("RGB")).astype(np.float32); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return ((g > r * 1.06) & (g > b * 1.12) & (g > 35) & (g < 170)), IMG[key]["extent"]
masks = {k: canopy(k) for k in ("near", "city")}
def frac(key, x, y, rad):
    m, e = masks[key]; H, W = m.shape; sx = W / (e[2] - e[0]); sy = H / (e[3] - e[1])
    cx = (x - e[0]) * sx; cy = (e[3] - y) * sy; rx = max(1, int(rad * sx)); ry = max(1, int(rad * sy))
    if cx < rx or cy < ry or cx >= W - rx or cy >= H - ry: return None
    return float(m[int(cy) - ry:int(cy) + ry + 1, int(cx) - rx:int(cx) + rx + 1].mean())
STEP = 6.5; tp = Point(tower["x"], tower["y"])
for ix in range(int((x1 - x0) / STEP)):
    for iy in range(int((y1 - y0) / STEP)):
        x = x0 + (ix + rnd.random()) * STEP; y = y0 + (iy + rnd.random()) * STEP
        f = frac("near", x, y, 2.2)
        if f is None: f = frac("city", x, y, 2.2)
        if f is None or f < 0.55: continue
        pt = Point(x, y)
        if psea.contains(pt) or pt.distance(tp) < 17 or any(polys[k].distance(pt) < 1.5 for k in btree.query(pt.buffer(2))): continue
        spots.append((x, y))
LEAF = float(os.environ.get("LEAF", "1"))   # < 1 thins the city trees for a lighter file
def foliage(mesh, cx, cy, cz, rx, rz, n, size, col, rnd, mat=0):
    """a cloud of small leaf triangles on an ellipsoid shell: gives a ragged, leafy outline instead of a smooth ball."""
    u = np.array([rnd.gauss(0, 1) for _ in range(3 * n)]).reshape(n, 3); u /= np.linalg.norm(u, axis=1)[:, None]
    rho = np.array([rnd.uniform(0.72, 1.08) for _ in range(n)])[:, None]; c = np.array([cx, cy, cz]) + u * rho * np.array([rx, rx, rz])
    d = np.array([rnd.gauss(0, 1) for _ in range(6 * n)]).reshape(n, 2, 3) * size
    v = np.stack([c - d[:, 0] * 0.5, c + d[:, 0] * 0.5 + d[:, 1] * 0.25, c + d[:, 1] * 0.9 + u * size * 0.4], axis=1).reshape(-1, 3)
    cols = np.clip(np.array(col)[None, :] * np.array([rnd.uniform(0.6, 1.45) for _ in range(n)])[:, None], 0, 1).repeat(3, axis=0)
    mesh.v.append(v); mesh.t.append(np.arange(3 * n).reshape(n, 3) + mesh.n); mesh.n += 3 * n
    mesh.uv.append(v[:, :2].copy()); mesh.col.append(cols); mesh.mat.append(np.full(n, mat, np.int32))
def prism(mesh, prof, c, u, hw, z0, col, mat):
    """side profile [(along, height)] extruded across the width: used for car bodies."""
    ux, uy = u; nx, ny = -uy, ux; k = len(prof)
    v = [(c[0] + ux * a + nx * sgn * hw, c[1] + uy * a + ny * sgn * hw, z0 + h) for sgn in (-1, 1) for a, h in prof]; t = []
    for i in range(k):
        j = (i + 1) % k; t += [(i, j, k + j), (i, k + j, k + i)]
    for i in range(1, k - 1): t += [(0, i + 1, i), (k, k + i, k + i + 1)]
    mesh.add(v, t, col=col, mat=mat)
def blob(cx, cy, cz, rx, rz, rnd, seg=7, rings=4):
    v = [(cx, cy, cz - rz)]
    for k in range(1, rings):
        ph = math.pi * k / rings
        for s in range(seg):
            th = 2 * math.pi * (s + 0.5 * (k % 2)) / seg; w = rnd.uniform(0.8, 1.15)
            v.append((cx + rx * w * math.sin(ph) * math.cos(th), cy + rx * w * math.sin(ph) * math.sin(th), cz - rz * math.cos(ph) * w))
    v.append((cx, cy, cz + rz)); t = []; top = len(v) - 1
    for s in range(seg): t.append((0, 1 + (s + 1) % seg, 1 + s)); t.append((top, top - seg + s, top - seg + (s + 1) % seg))
    for k in range(rings - 2):
        o = 1 + k * seg
        for s in range(seg): t += [(o + s, o + (s + 1) % seg, o + seg + (s + 1) % seg), (o + s, o + seg + (s + 1) % seg, o + seg + s)]
    return v, t
near_spots = [p for p in spots if math.hypot(p[0] - TX, p[1] - TY) < 190]; spots = [p for p in spots if math.hypot(p[0] - TX, p[1] - TY) >= 190]
WEB_TREES = []; WEB_CLUMPS = []
tm = Mesh()
for (x, y) in spots:
    z = land_h(x, y); hgt = rnd.uniform(6, 11); r = hgt * rnd.uniform(0.26, 0.34)
    gcol = (rnd.uniform(0.07, 0.13), rnd.uniform(0.15, 0.24), rnd.uniform(0.04, 0.08))
    tr = [(x + 0.18 * math.cos(k * 2.094), y + 0.18 * math.sin(k * 2.094), zz) for zz in (z, z + hgt * 0.45) for k in range(3)]
    tm.add(tr, [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)], col=(0.16, 0.11, 0.07), mat=0)
    WEB_TREES.append([round(x, 2), round(y, 2), round(z, 2), round(hgt, 2), round(r, 2)] + [round(c, 3) for c in gcol])
    close = math.hypot(x - TX, y - TY) < 520; sg, rg = (10, 6) if close else (6, 4)
    for k in range(4 if close else 2):
        off = 0 if k == 0 else 0.55; kk = 1 if k == 0 else rnd.uniform(0.55, 0.8); gc = tuple(c * rnd.uniform(0.85, 1.15) for c in gcol)
        bx, by, bz = x + rnd.uniform(-1, 1) * r * off, y + rnd.uniform(-1, 1) * r * off, z + hgt * (0.68 if k == 0 else rnd.uniform(0.5, 0.75))
        sh = 0.88 if close else 1.0
        v, t = blob(bx, by, bz, r * kk * sh, hgt * 0.34 * kk * sh, rnd, sg, rg); tm.add(v, t, col=tuple(c * 0.55 for c in gc) if close else gc, mat=0)
        if close: foliage(tm, bx, by, bz, r * kk, hgt * 0.34 * kk, int(420 * LEAF), 0.32 / LEAF ** 0.5, gc, rnd)
tm.save("trees"); print("trees", len(spots))

# ================= the tower's surroundings (within 260 m) =================
RS = 260.0; disc = Point(TX, TY).buffer(RS, 64); land = disc.difference(sea); S = Mesh()
GRASS, PAVE, COBBLE, ASPHALT, CURB, PROM, LAMP, GLOW, CAR, CARGLASS, BARK, PINE, MARK, WALK, TERRA, MARBLE, DIRT, STELE = range(18)
def flat(poly, z, mat, col=(1, 1, 1)):
    for pp in (poly.geoms if hasattr(poly, "geoms") else [poly]):
        if pp.geom_type == "Polygon" and pp.area > 0.3:
            v, t = tri(pp); S.add(np.c_[v, np.full(len(v), FLAT + z)], t, col=col, mat=mat)
bu = unary_union([p for p in polys if p.distance(TW) < RS + 30])
from shapely.geometry import LineString
roads = [r for r in L["roads"] if LineString(r["line"]).distance(TW) < RS]
drive = unary_union([LineString(r["line"]).buffer(r["w"] / 2, cap_style=1) for r in roads if r["kind"] in ("primary", "secondary", "tertiary", "residential", "unclassified", "trunk", "service", "living_street")]).intersection(land)
gl = [(Polygon(p[0], p[1:]).buffer(0), it.get("name") or "") for it in L["green"] for p in it["poly"]]
plaza = unary_union([g for g, n in gl if "Πλατεία" in n and g.distance(TW) < 5])                  # the square's outline: paving, not lawn
greens = [g for g, n in gl if g.distance(TW) < RS and "Πλατεία" not in n and g.distance(TW) > 13 and not g.is_empty]
# the tower's own lawn, read off the aerial photo: a horseshoe of stone panels 24-31 m out on the sea side, open towards the road
R_IN, R_OUT, NP, SPAN, R_BACK = 24.0, 31.0, 10, math.radians(220), 17.5
A0 = math.radians(225) - SPAN / 2; ANG = [A0 + SPAN * i / NP for i in range(NP + 1)]
def rp(r, a): return (TX + r * math.cos(a), TY + r * math.sin(a))
back = [rp(R_BACK, A0 + SPAN + (2 * math.pi - SPAN) * i / 12) for i in range(1, 12)]
lawn = Polygon([rp(R_IN - 2.4, a) for a in ANG] + back); gravel = Polygon([rp(R_IN, a) for a in ANG] + [rp(R_IN - 2.4, a) for a in ANG[::-1]])
ringpoly = Polygon([rp(R_OUT, a) for a in ANG] + [rp(R_IN, a) for a in ANG[::-1]])
lawn = lawn.difference(drive.buffer(3.0)); ringpoly = ringpoly.difference(drive.buffer(3.0))
walk = drive.buffer(2.6, join_style=1).difference(drive).intersection(land).difference(bu.buffer(0.1))      # raised pavements beside every road
grass = unary_union(greens).intersection(land).difference(drive.buffer(2.6)).difference(lawn.buffer(8))
prom = sea.buffer(21).intersection(land).difference(grass).difference(drive)
terra = plaza.intersection(land).difference(prom).difference(drive.buffer(2.6)) if not plaza.is_empty else Polygon()
flat(land.difference(bu.buffer(0.2)), 0.02, PAVE)
flat(terra, 0.03, TERRA)
flat(prom, 0.04, PROM)
flat(sea.buffer(1.1).intersection(land), 0.05, CURB)                                              # stone coping along the quay
flat(drive, 0.06, ASPHALT)
flat(walk, 0.20, WALK)
for pp in (walk.geoms if hasattr(walk, "geoms") else [walk]):                                    # kerb faces
    if pp.geom_type != "Polygon": continue
    for ring in [pp.exterior] + list(pp.interiors):
        c = list(ring.coords)
        for p, r in zip(c[:-1], c[1:]): S.add([(p[0], p[1], FLAT + .02), (r[0], r[1], FLAT + .02), (r[0], r[1], FLAT + .2), (p[0], p[1], FLAT + .2)], Q, mat=CURB)
# ---- road markings: lane lines, edge lines, zebra crossings ----
DRIVE = ("primary", "secondary", "tertiary", "residential", "unclassified", "trunk", "living_street")
rl = [(r, LineString(r["line"])) for r in roads if r["kind"] in DRIVE and LineString(r["line"]).length > 6]
def junction(pt, me):
    k = 0
    for r2, l2 in rl:
        if l2 is me or l2.distance(pt) > 0.4: continue
        s2 = l2.project(pt); k += 2 if 1.0 < s2 < l2.length - 1.0 else 1
    return k >= 2
JUNC = []; ENDS = {}
for r, ls in rl:
    ENDS[id(ls)] = [junction(Point(ls.coords[0]), ls), junction(Point(ls.coords[-1]), ls)]
    JUNC += [ls.coords[k] for k, f in zip((0, -1), ENDS[id(ls)]) if f]
JUNC = np.array(JUNC) if JUNC else np.zeros((0, 2))
def jdist(x, y): return float(np.hypot(JUNC[:, 0] - x, JUNC[:, 1] - y).min()) if len(JUNC) else 1e9
ZM = FLAT + 0.075; pland = prep(land)
def lane_info(r):
    w = r["w"]; one = r.get("oneway")
    n = r.get("lanes") or (max(1, round(w / 3.4)) if one else max(2, 2 * round(w / 6.8)))
    if not one and n % 2: n += 1
    return w, one, n, w / n
def paint(ls, off, s0, s1, wd=0.15):
    a, b = ls.interpolate(s0), ls.interpolate(s1); dx, dy = b.x - a.x, b.y - a.y; n = math.hypot(dx, dy)
    if n < 0.2: return
    lx, ly = -dy / n, dx / n; ax, ay, bx_, by_ = a.x + lx * off, a.y + ly * off, b.x + lx * off, b.y + ly * off
    if not pland.contains(Point((ax + bx_) / 2, (ay + by_) / 2)): return
    hx, hy = lx * wd / 2, ly * wd / 2
    S.add([(ax - hx, ay - hy, ZM), (bx_ - hx, by_ - hy, ZM), (bx_ + hx, by_ + hy, ZM), (ax + hx, ay + hy, ZM)], Q, mat=MARK)
def stripe(ls, off, dash, gap, s0, s1):
    s = s0
    while s < s1 - 0.5:
        e = min(s + dash, s1); t = s
        while t < e - 1e-6: t2 = min(t + 2.0, e); paint(ls, off, t, t2); t = t2
        s = e + gap
ncross = 0
for r, ls in rl:
    w, one, n, lw = lane_info(r); j0, j1 = ENDS[id(ls)]; JW = w / 2 + 7.5; s0 = JW if j0 else 0.0; s1 = ls.length - (JW if j1 else 0.0)
    if s1 - s0 > 3:
        if r["kind"] != "residential":
            for o in (-(w / 2 - 0.25), w / 2 - 0.25): stripe(ls, o, 1e9, 0, s0, s1)
        for k in range(1, n):
            o = -w / 2 + k * lw
            if not one and k == n // 2:
                if n >= 4: stripe(ls, o - 0.12, 1e9, 0, s0, s1); stripe(ls, o + 0.12, 1e9, 0, s0, s1)
                else: stripe(ls, o, 3.0, 4.5, s0, s1)
            else: stripe(ls, o, 3.0, 6.0, s0, s1)
    if ls.length > 22:
        for flag, sc in ((j0, w / 2 + 3.2), (j1, ls.length - w / 2 - 3.2)):
            if not flag: continue
            a, b = ls.interpolate(sc - 0.5), ls.interpolate(sc + 0.5); ux, uy = b.x - a.x, b.y - a.y; q = ls.interpolate(sc); ncross += 1
            for i in range(int((w - 0.6) / 1.0)):
                o = -w / 2 + 0.55 + i * 1.0; c = (q.x - uy * o, q.y + ux * o)
                if pland.contains(Point(c)): obox(S, c, (ux, uy), 3.0, 0.5, ZM - 0.005, ZM, (1, 1, 1), MARK)
def wallring(poly, z0, z1, mat):
    for pp in (poly.geoms if hasattr(poly, "geoms") else [poly]):
        if pp.geom_type != "Polygon": continue
        for ring in [pp.exterior] + list(pp.interiors):
            c = list(ring.coords)
            for p, r in zip(c[:-1], c[1:]): S.add([(p[0], p[1], FLAT + z0), (r[0], r[1], FLAT + z0), (r[0], r[1], FLAT + z1), (p[0], p[1], FLAT + z1)], Q, mat=mat)
nplant = 0; ndec = 0; drnd = random.Random(21)
for g in (grass.geoms if hasattr(grass, "geoms") else [grass]):
    if g.geom_type != "Polygon" or g.area < 8: continue
    if g.area < 1300:                                                       # raised planter with a white marble seat wall
        inner = g.buffer(-0.4, join_style=2); nplant += 1
        if inner.is_empty: continue
        flat(inner, 0.42, GRASS); rim = g.difference(inner); flat(rim, 0.5, MARBLE); wallring(rim, 0.02, 0.5, MARBLE)
        cx, cy = g.centroid.x - TX, g.centroid.y - TY
        if -30 < cx < 30 and -118 < cy < -15 and g.area < 700:               # the long planters of the strip carry a row of small deciduous trees
            c = list(g.minimum_rotated_rectangle.exterior.coords); d, p, r = max((math.dist(c[i], c[i + 1]), c[i], c[i + 1]) for i in range(4))
            ux, uy = (r[0] - p[0]) / d, (r[1] - p[1]) / d; m0 = g.centroid
            for i in range(int(d / 5.5) + 1):
                o = (i - int(d / 5.5) / 2) * 5.5; x, y = m0.x + ux * o + drnd.uniform(-.4, .4), m0.y + uy * o + drnd.uniform(-.4, .4)
                if not inner.contains(Point(x, y)): continue
                H = drnd.uniform(4.6, 5.8); z = FLAT + 0.42; col = (drnd.uniform(0.16, 0.24), drnd.uniform(0.30, 0.40), drnd.uniform(0.05, 0.09)); ndec += 1
                obox(S, (x, y), (1, 0), 0.22, 0.22, z, z + H * 0.5, (1, 1, 1), BARK)
                for k in range(4):
                    a = drnd.uniform(0, 6.28); dd = 0 if k == 0 else drnd.uniform(0.5, 1.1); bx, by, bz = x + dd * math.cos(a), y + dd * math.sin(a), z + H * drnd.uniform(0.62, 0.8); rx = drnd.uniform(1.3, 1.9)
                    vv, tt = blob(bx, by, bz, rx * 0.8, rx * 0.6, drnd, 9, 5); S.add(vv, tt, col=tuple(k2 * 0.6 for k2 in col), mat=PINE)
                    foliage(S, bx, by, bz, rx, rx * 0.75, 380, 0.26, col, drnd, PINE); WEB_CLUMPS.append([round(bx, 2), round(by, 2), round(bz, 2), round(rx, 2), round(rx * 0.75, 2)] + [round(k2, 3) for k2 in col])
    else:                                                                    # big lawns lie flush with a thin kerb
        flat(g, 0.10, GRASS); flat(g.boundary.buffer(0.2).intersection(land), 0.16, CURB)
# the tower lawn: grass, a beaten path round the base, two paths out to the ring, a paved way to the door
DOOR = math.radians(85); tdisc = Point(TX, TY).buffer(11.9, 48)
basepath = Point(TX, TY).buffer(13.8, 48).difference(tdisc)
spokes = unary_union([LineString([rp(13, a), rp(R_IN, a)]).buffer(0.9) for a in (math.radians(190), math.radians(265))])
doorway = LineString([rp(11.5, DOOR), rp(R_BACK + 6, DOOR)]).buffer(2.0, cap_style=2)
dirt = basepath.union(spokes).intersection(lawn).difference(doorway)
flat(lawn.difference(tdisc).difference(dirt).difference(doorway), 0.10, GRASS); flat(dirt, 0.09, DIRT); flat(doorway.difference(tdisc).intersection(land).difference(drive.buffer(2.6)), 0.11, PROM)
backarc = LineString([rp(R_BACK, A0 + SPAN)] + back + [rp(R_BACK, A0)]).buffer(0.3).difference(doorway.buffer(0.3)).difference(drive.buffer(3.0))   # low wall on the road side
flat(backarc, 0.85, CURB); wallring(backarc, 0.02, 0.85, CURB)
# the horseshoe: raised rubble-stone slabs, high at the outer edge and sloping down towards the lawn, on a bed of gravel
flat(ringpoly, 0.06, DIRT); flat(gravel.difference(drive.buffer(3.0)).difference(doorway), 0.08, DIRT); panels = 0
def slab_h(q): return 0.14 + 0.5 * (math.hypot(q[0] - TX, q[1] - TY) - R_IN) / (R_OUT - R_IN)
for i in range(NP):
    gp = math.radians(1.6); a0, a1 = ANG[i] + gp, ANG[i + 1] - gp
    pp = Polygon([rp(R_IN + 0.3, a0), rp(R_OUT - 0.3, a0), rp(R_OUT - 0.3, a1), rp(R_IN + 0.3, a1)]).intersection(ringpoly)
    if pp.is_empty or pp.geom_type != "Polygon" or pp.area < 15: continue
    pp = pp.segmentize(1.5); v, t = tri(pp); S.add(np.c_[v, FLAT + np.array([slab_h(q) for q in v])], t, mat=COBBLE); panels += 1
    c = list(pp.exterior.coords)
    for p_, r_ in zip(c[:-1], c[1:]):
        S.add([(p_[0], p_[1], FLAT + 0.04), (r_[0], r_[1], FLAT + 0.04), (r_[0], r_[1], FLAT + slab_h(r_)), (p_[0], p_[1], FLAT + slab_h(p_))], Q, uv=[(0, 0), (math.dist(p_, r_), 0), (math.dist(p_, r_), slab_h(r_)), (0, slab_h(p_))], mat=COBBLE)
# one line of tall stone stelae along the edge of the big lawn, with low marble benches in front (positions read off the aerial photo)
P0 = (TX + 47, TY - 27); P1 = (TX - 15, TY - 113); LN = math.dist(P0, P1); sx, sy = (P1[0] - P0[0]) / LN, (P1[1] - P0[1]) / LN; wx, wy = sy, -sx    # w points to the walk (west)
nstele = 0; srnd = random.Random(4)
def stele(c, u, z):
    ux, uy = u; nx, ny = -uy, ux; hb, ht, tb, tt, H = 0.58, 0.44, 0.24, 0.17, srnd.uniform(1.65, 1.85)
    b = [(c[0] + a * ux + d * nx, c[1] + a * uy + d * ny, z + h) for h, hw, th in ((0, hb, tb), (H, ht, tt)) for a, d in ((-hw, -th), (hw, -th), (hw, th), (-hw, th))]
    S.add(b, [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)], mat=STELE)
for i in range(int(LN / 2.55) + 1):
    q = (P0[0] + sx * i * 2.55, P0[1] + sy * i * 2.55)
    if pland.contains(Point(q)): stele(q, (sx, sy), FLAT + 0.02); nstele += 1
for i in range(int(LN / 11.5)):
    m = 6 + i * 11.5; q = (P0[0] + sx * m + wx * 4.3, P0[1] + sy * m + wy * 4.3)
    if pland.contains(Point(q)): obox(S, q, (sx, sy), 5.2, 0.55, FLAT + 0.02, FLAT + 0.45, (1, 1, 1), MARBLE)
# lamp posts along the promenade
coastl = sea.buffer(4.5).boundary.intersection(land.buffer(-1)); nl = 0
for ls in (coastl.geoms if hasattr(coastl, "geoms") else [coastl]):
    if ls.geom_type != "LineString": continue
    for i in range(int(ls.length / 26)):
        q = ls.interpolate(13 + i * 26); x, y = q.x, q.y; nl += 1
        obox(S, (x, y), (1, 0), 0.45, 0.45, FLAT, FLAT + 0.7, (1, 1, 1), LAMP); obox(S, (x, y), (1, 0), 0.14, 0.14, FLAT + 0.7, FLAT + 6.2, (1, 1, 1), LAMP)
        obox(S, (x, y), (1, 0), 0.9, 0.08, FLAT + 5.9, FLAT + 6.0, (1, 1, 1), LAMP)
        for sx in (-0.45, 0.45): obox(S, (x + sx, y), (1, 0), 0.36, 0.36, FLAT + 6.0, FLAT + 6.55, (1, 1, 1), GLOW); obox(S, (x + sx, y), (1, 0), 0.46, 0.46, FLAT + 6.55, FLAT + 6.65, (1, 1, 1), LAMP)
# ---- cars: in lanes, facing the way the traffic flows, parked along the kerb in side streets ----
CARCOL = [(0.8, 0.8, 0.8), (0.75, 0.75, 0.77), (0.05, 0.05, 0.06), (0.3, 0.31, 0.33), (0.5, 0.04, 0.04), (0.06, 0.12, 0.35), (0.55, 0.56, 0.58), (0.85, 0.85, 0.83), (0.6, 0.6, 0.62), (0.12, 0.13, 0.15)]
crnd = random.Random(11); ncar = 0; placed = []
def car(c, ux, uy):
    col = CARCOL[crnd.randrange(len(CARCOL))]; ln = crnd.uniform(3.9, 4.6)
    body = [(-.5, .30), (-.5, .62), (-.46, .76), (-.22, .82), (.20, .84), (.46, .72), (.5, .52), (.5, .30)]
    cab = [(-.37, .80), (-.27, 1.30), (.08, 1.33), (.25, .83)]; van = crnd.random() < 0.1
    if van: body = [(-.5, .30), (-.5, 1.75), (.30, 1.78), (.42, 1.25), (.5, .8), (.5, .30)]; cab = [(.26, 1.2), (.31, 1.7), (.33, 1.7), (.44, 1.2)]; col = (0.82, 0.82, 0.8)
    Z = FLAT + 0.06
    prism(S, [(a * ln, h) for a, h in body], c, (ux, uy), 0.88, Z, col, CAR)
    prism(S, [(a * ln, h) for a, h in cab], c, (ux, uy), 0.80 if not van else 0.86, Z, (1, 1, 1), CARGLASS)
    if not van: prism(S, [(-.27 * ln, 1.29), (-.27 * ln, 1.34), (.08 * ln, 1.37), (.08 * ln, 1.32)], c, (ux, uy), 0.81, Z, col, CAR)
    for a in (-.31, .31):
        for sd in (-1, 1): obox(S, (c[0] + ux * a * ln - uy * sd * 0.80, c[1] + uy * a * ln + ux * sd * 0.80), (ux, uy), 0.64, 0.2, Z + 0.02, Z + 0.64, (1, 1, 1), CARGLASS)
def lane(ls, off, forward, lo, hi, clear, skip):
    """drop cars along one lane: off = metres to the left of the line, spacing lo..hi, kept `clear` metres from junctions."""
    global ncar
    d = crnd.uniform(2, hi)
    while d < ls.length - 2.5:
        a, b = ls.interpolate(d - 0.5), ls.interpolate(d + 0.5); ux, uy = b.x - a.x, b.y - a.y; n = math.hypot(ux, uy) or 1; ux, uy = ux / n, uy / n
        q = ls.interpolate(d); c = (q.x - uy * off, q.y + ux * off); d += crnd.uniform(lo, hi)
        if crnd.random() < skip or jdist(*c) < clear or not pland.contains(Point(c)) or bu.distance(Point(c)) < 1.3: continue
        if any(math.hypot(c[0] - px, c[1] - py) < 4.9 for px, py in placed): continue
        placed.append(c); ncar += 1; car(c, ux, uy) if forward else car(c, -ux, -uy)
for r, ls in rl:
    w, one, n, lw = lane_info(r)
    if r["kind"] in ("residential", "unclassified", "living_street") and w <= 8.5:
        for o in (-(w / 2 - 1.05), w / 2 - 1.05): lane(ls, o, one or o < 0, 5.3, 6.2, 7.0, 0.22)      # parked nose to tail
        lane(ls, 0.0 if one else -0.9, True, 25, 60, 8.0, 0.3)
    else:
        for k in range(n):
            o = -w / 2 + (k + 0.5) * lw; lane(ls, o, one or o < 0, 9, 26, w / 2 + 4.0, 0.15)
# stone pines: bare leaning trunk, broad flat crown of several clumps
prnd = random.Random(5); pines = []
for (x, y) in sorted(near_spots, key=lambda p: prnd.random()):
    if all(math.hypot(x - a, y - b) > 6.5 for a, b in pines) and not drive.buffer(1).contains(Point(x, y)) and not ringpoly.buffer(0.8).contains(Point(x, y)) and not doorway.contains(Point(x, y)): pines.append((x, y))
for (x, y) in pines:
    big = lawn.buffer(25).contains(Point(x, y)); H = prnd.uniform(8, 12) if big else prnd.uniform(5, 9); cr = H * prnd.uniform(0.38, 0.5)
    lean = prnd.uniform(0, 2 * math.pi); lx, ly = math.cos(lean) * H * 0.12, math.sin(lean) * H * 0.12; K = 5; rings = []
    for i in range(K + 1):
        f = i / K; cx, cy, cz = x + lx * f * f, y + ly * f * f, FLAT + H * 0.72 * f; rr = 0.28 * (1 - 0.45 * f) * (H / 9)
        rings.append([(cx + rr * math.cos(j * math.pi / 3), cy + rr * math.sin(j * math.pi / 3), cz) for j in range(6)])
    v = [q for ring in rings for q in ring]; t = []
    for i in range(K):
        for j in range(6): a = i * 6 + j; b = i * 6 + (j + 1) % 6; t += [(a, b, b + 6), (a, b + 6, a + 6)]
    S.add(v, t, col=(1, 1, 1), mat=BARK); tx_, ty_, tz_ = x + lx, y + ly, FLAT + H * 0.72
    g = (prnd.uniform(0.035, 0.06), prnd.uniform(0.09, 0.14), prnd.uniform(0.025, 0.045))
    for k in range(prnd.randint(6, 9)):
        a = prnd.uniform(0, 2 * math.pi); d = cr * prnd.uniform(0.0, 0.72) if k else 0; rx = cr * prnd.uniform(0.38, 0.6)
        bx, by, bz = tx_ + d * math.cos(a), ty_ + d * math.sin(a), tz_ + H * prnd.uniform(0.06, 0.2)
        rzz = rx * prnd.uniform(0.42, 0.58); gk = tuple(c * prnd.uniform(0.8, 1.25) for c in g)
        vv, tt = blob(bx, by, bz, rx * 0.86, rzz * 0.86, prnd, 12, 7); S.add(vv, tt, col=tuple(c * 0.45 for c in gk), mat=PINE)
        foliage(S, bx, by, bz + rzz * 0.15, rx, rzz, 1100, 0.26, gk, prnd, PINE)
        WEB_CLUMPS.append([round(bx, 2), round(by, 2), round(bz + rzz * 0.15, 2), round(rx, 2), round(rzz, 2)] + [round(c, 3) for c in gk])
        a0 = np.array([tx_, ty_, tz_ - 0.8]); b0 = np.array([bx, by, bz - rzz * 0.4]); sd = np.cross(b0 - a0, (0, 0, 1.0)); sd = sd / (np.linalg.norm(sd) + 1e-6) * 0.09
        S.add([a0 - sd, a0 + sd, b0 + sd * 0.5, b0 - sd * 0.5, a0 + (0, 0, 0.16), b0 + (0, 0, 0.1)], [(0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 4), (1, 4, 5), (1, 5, 2)], col=(1, 1, 1), mat=BARK)
S.save("site"); print("crossings", ncross, "junctions", len(JUNC)); print("planters", nplant, "small trees", ndec, "stelae", nstele); print("site: panels", panels, "lamps", nl, "cars", ncar, "pines", len(pines), "lawn m2", round(lawn.area))

json.dump({"trees": WEB_TREES, "clumps": WEB_CLUMPS, "tower": [TX, TY, FLAT], "bounds": [x0, y0, x1, y1]}, open(f"{root}/data/web_trees.json", "w"))
