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
_pl = {}
for it in L.get("plaza", []):
    if it.get("name") in ("Πλατεία Αριστοτέλους", "Πεζόδρομος Αριστοτέλους"):
        _pl[it["name"]] = unary_union([Polygon(p_[0], p_[1:]).buffer(0) for p_ in it["poly"]] + ([_pl[it["name"]]] if it["name"] in _pl else []))
ASQ = _pl["Πλατεία Αριστοτέλους"]; AC = ASQ.centroid; ACX, ACY = AC.x, AC.y
APLZ = unary_union(list(_pl.values())).intersection(AC.buffer(330))
_c = list(ASQ.minimum_rotated_rectangle.exterior.coords); _e1 = np.array(_c[1]) - np.array(_c[0]); _e2 = np.array(_c[2]) - np.array(_c[1]); _g = np.array([0.62, 0.78])
av, tv = (_e1, _e2) if abs(_e1 @ _g) / np.linalg.norm(_e1) > abs(_e2 @ _g) / np.linalg.norm(_e2) else (_e2, _e1)
La, Wt = float(np.linalg.norm(av)), float(np.linalg.norm(tv)); av = av / La * (1 if av @ _g > 0 else -1); tv = np.array([av[1], -av[0]]); M0 = np.array(ASQ.minimum_rotated_rectangle.centroid.coords[0])
def Wd(al, tl): q = M0 + av * al + tv * tl; return (float(q[0]), float(q[1]))      # square coordinates: al = metres inland from its centre, tl = metres to the east
def land_h(x, y):
    h = max(bil(Zi, (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)), 0.0) + QUAY
    w = min(max((math.hypot(x - TX, y - TY) - 270) / 150, 0), 1); w = w * w * (3 - 2 * w)   # the real plaza is level
    w2 = min(max((math.hypot(x - ACX, y - ACY) - 340) / 120, 0), 1); w2 = w2 * w2 * (3 - 2 * w2); w = min(w, w2)
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
def limb(mesh, p0, p1, r0, r1, col, mat, n=7, cap=True):
    """tapered cylinder between two points."""
    p0 = np.array(p0, float); p1 = np.array(p1, float); d = p1 - p0; ln = np.linalg.norm(d)
    if ln < 1e-6: return
    d /= ln; ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0]); e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    ang = np.arange(n) * 2 * np.pi / n; ring = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    v = np.vstack([p0 + ring * r0, p1 + ring * r1, [p0], [p1]]); t = []
    for i in range(n):
        j = (i + 1) % n; t += [(i, j, n + j), (i, n + j, n + i)]
        if cap: t += [(2 * n, j, i), (2 * n + 1, n + i, n + j)]
    mesh.add(v, t, col=col, mat=mat)
P_SKIN = [(0.78, 0.60, 0.48), (0.70, 0.52, 0.40), (0.85, 0.68, 0.56), (0.55, 0.38, 0.28)]; P_HAIR = [(0.05, 0.04, 0.03), (0.16, 0.10, 0.05), (0.30, 0.20, 0.10), (0.55, 0.52, 0.50), (0.45, 0.32, 0.12)]
P_SHIRT = [(0.85, 0.85, 0.83), (0.10, 0.10, 0.12), (0.65, 0.12, 0.10), (0.12, 0.25, 0.50), (0.80, 0.70, 0.20), (0.20, 0.42, 0.30), (0.55, 0.55, 0.58), (0.86, 0.50, 0.20), (0.45, 0.25, 0.45), (0.75, 0.80, 0.88)]
P_PANTS = [(0.10, 0.12, 0.20), (0.08, 0.08, 0.09), (0.30, 0.32, 0.38), (0.45, 0.38, 0.28), (0.15, 0.20, 0.32), (0.60, 0.58, 0.52)]
def person2(mesh, x, y, z, a, k, rnd, mat):
    """a walking figure built from tapered limbs: legs in stride, arms swinging against them, torso, neck, head and hair."""
    f = np.array([math.cos(a), math.sin(a), 0.0]); sd = np.array([-math.sin(a), math.cos(a), 0.0]); O = np.array([x, y, z]); zu = np.array([0, 0, 1.0])
    def Pt(fw, side, h): return O + (f * fw + sd * side) * k + zu * h * k
    sk = P_SKIN[rnd.randrange(4)]; sh = P_SHIRT[rnd.randrange(len(P_SHIRT))]; pa = P_PANTS[rnd.randrange(len(P_PANTS))]; hr = P_HAIR[rnd.randrange(5)]; short = rnd.random() < 0.6; st = rnd.uniform(-0.3, 0.3)
    for sg in (-1, 1):
        hip = Pt(0, sg * 0.085, 0.90); ank = Pt(sg * st * 0.5, sg * 0.085, 0.07); knee = (hip + ank) / 2 + f * 0.05 * k
        limb(mesh, hip, knee, 0.085 * k, 0.062 * k, pa, mat); limb(mesh, knee, ank, 0.06 * k, 0.042 * k, pa, mat); limb(mesh, ank - zu * 0.03 * k, ank + f * 0.2 * k - zu * 0.04 * k, 0.045 * k, 0.038 * k, (0.06, 0.05, 0.05), mat, 6)
        s0 = Pt(0, sg * 0.2, 1.42); el = Pt(-sg * st * 0.22, sg * 0.235, 1.14); hd = Pt(-sg * st * 0.42 + 0.03, sg * 0.23, 0.88)
        limb(mesh, s0, el, 0.05 * k, 0.042 * k, sh, mat); limb(mesh, el, hd, 0.04 * k, 0.03 * k, sk if short else sh, mat)
    limb(mesh, Pt(0, 0, 0.84), Pt(0, 0, 1.0), 0.15 * k, 0.15 * k, pa, mat, 9); limb(mesh, Pt(0, 0, 1.0), Pt(0, 0, 1.45), 0.15 * k, 0.185 * k, sh, mat, 9); limb(mesh, Pt(0, 0, 1.45), Pt(0, 0, 1.54), 0.05 * k, 0.05 * k, sk, mat, 6)
    c = Pt(0.01, 0, 1.64); v, t = blob(c[0], c[1], c[2], 0.10 * k, 0.12 * k, rnd, 9, 6); mesh.add(v, t, col=sk, mat=mat)
    c = Pt(-0.025, 0, 1.675); v, t = blob(c[0], c[1], c[2], 0.105 * k, 0.10 * k, rnd, 9, 5); mesh.add(v, t, col=hr, mat=mat)
def palm(mesh, x, y, z, Ht, rnd, mcol, mleaf):
    """date palm: thick trunk, crown of arching fronds with drooping leaflets."""
    a = rnd.uniform(0, 6.28); lx, ly = math.cos(a) * 0.05 * Ht, math.sin(a) * 0.05 * Ht; prev = np.array([x, y, z]); tc = (0.26, 0.20, 0.14)
    for i in range(1, 6):
        fq = i / 5; cur = np.array([x + lx * fq * fq, y + ly * fq * fq, z + Ht * fq]); limb(mesh, prev, cur, 0.30 - 0.06 * (i - 1) / 5, 0.30 - 0.06 * fq, tuple(c * rnd.uniform(0.85, 1.1) for c in tc), mcol, 9, False); prev = cur
    v, t = blob(prev[0], prev[1], prev[2] + 0.1, 0.5, 0.55, rnd, 9, 6); mesh.add(v, t, col=(0.30, 0.22, 0.12), mat=mcol); zu = np.array([0, 0, 1.0]); V = []; T = []; C = []
    for i in range(30):
        az = rnd.uniform(0, 6.28); th = rnd.uniform(-0.45, 1.35); Lf = rnd.uniform(2.9, 4.0); dh = np.array([math.cos(az), math.sin(az), 0]); sdv = np.array([-math.sin(az), math.cos(az), 0]); pos = prev + zu * 0.3; pts = []
        for j in range(8): pts.append(pos.copy()); pos = pos + (Lf / 7) * (math.cos(th) * dh + math.sin(th) * zu); th -= 0.26 + 0.05 * j
        g = (rnd.uniform(0.06, 0.11), rnd.uniform(0.17, 0.27), rnd.uniform(0.04, 0.07))
        for j in range(7):
            w0 = 0.55 * math.sin(math.pi * min(1, j / 7 * 1.05 + 0.08)) ** 0.7; w1 = 0.55 * math.sin(math.pi * min(1, (j + 1) / 7 * 1.05 + 0.08)) ** 0.7 if j < 6 else 0.03; c0, c1 = pts[j], pts[j + 1]; b = len(V)
            V += [c0, c1, c0 + sdv * w0 - zu * w0 * 0.4, c1 + sdv * w1 - zu * w1 * 0.4, c0 - sdv * w0 - zu * w0 * 0.4, c1 - sdv * w1 - zu * w1 * 0.4]; T += [(b, b + 1, b + 3), (b, b + 3, b + 2), (b, b + 4, b + 5), (b, b + 5, b + 1)]; C += [g] * 6
    mesh.v.append(np.array(V)); mesh.t.append(np.array(T) + mesh.n); mesh.n += len(V); mesh.uv.append(np.array(V)[:, :2].copy()); mesh.col.append(np.array(C)); mesh.mat.append(np.full(len(T), mleaf, np.int32))
def chair(mesh, q, u, z, col, mat):
    n = (-u[1], u[0]); obox(mesh, q, u, 0.42, 0.42, z + 0.42, z + 0.46, col, mat); obox(mesh, (q[0] - u[0] * 0.2, q[1] - u[1] * 0.2), u, 0.04, 0.42, z + 0.46, z + 0.86, col, mat)
    for sa in (-0.18, 0.18):
        for sb in (-0.18, 0.18): obox(mesh, (q[0] + u[0] * sa + n[0] * sb, q[1] + u[1] * sa + n[1] * sb), u, 0.035, 0.035, z, z + 0.42, (0.12, 0.12, 0.12), mat)
# ---------- buildings ----------
PAL = [(0.80, 0.76, 0.68), (0.72, 0.68, 0.60), (0.84, 0.81, 0.75), (0.70, 0.63, 0.52), (0.78, 0.70, 0.56), (0.62, 0.61, 0.58), (0.82, 0.72, 0.60), (0.72, 0.56, 0.44), (0.86, 0.84, 0.80), (0.76, 0.66, 0.50), (0.66, 0.60, 0.54), (0.80, 0.62, 0.50)]
NEARPAL = [(0.74, 0.70, 0.62), (0.72, 0.62, 0.44), (0.68, 0.54, 0.36), (0.62, 0.60, 0.56), (0.78, 0.68, 0.52), (0.70, 0.48, 0.34), (0.60, 0.55, 0.44), (0.80, 0.77, 0.72), (0.66, 0.58, 0.40), (0.54, 0.56, 0.58)]
ROOF = [(0.55, 0.54, 0.52), (0.62, 0.60, 0.57), (0.48, 0.47, 0.46), (0.66, 0.63, 0.58)]
FABRIC = [(0.08, 0.26, 0.14), (0.70, 0.58, 0.38), (0.70, 0.30, 0.08), (0.10, 0.18, 0.40), (0.50, 0.09, 0.08), (0.80, 0.77, 0.68), (0.20, 0.36, 0.30)]
WALLM, ROOFM, PLAIN, GLASS = 0, 1, 2, 3
THEATRE_KTHBE, THEATRE_ROYAL, OFFICERS = "19790720", "19855083", "182044285"
bm = Mesh(); polys = []; tower = None; INNER = 420.0; LABELS = []
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
def polykatoikia(P, bd, zg, zt, wall, rnd, lite=False):
    """Greek apartment block: balconies on the street sides, awnings, set-back top floor, roof clutter."""
    body = P; top = zt
    if bd["h"] > 17 and rnd.random() < 0.65:
        inner = P.buffer(-rnd.uniform(1.8, 2.6), join_style=2)
        if inner.geom_type == "Polygon" and inner.area > 0.45 * P.area:
            top = zt - 3.1; walls(inner, top, zt, zg, wall, WALLM, rnd); roof(inner, zt, ROOF[rnd.randrange(4)]); parapet(inner, zt, wall, 0.5); body = inner
    walls(P, zg - 1.5, top, zg, wall, WALLM, rnd); roof(P, top, ROOF[rnd.randrange(4)]); parapet(P, top, wall, 0.95 if body is not P else 0.8)
    ext = P.exterior if P.exterior.is_ccw else P.exterior.reverse(); c = list(ext.coords); nfl = int((top - zg - 3.6) / 3.1 + 0.3)
    white = tuple(min(1, k * 1.04 + 0.02) for k in wall); deep = rnd.uniform(1.2, 1.7); style = rnd.choice(["solid", "solid", "rail", "segment", "segment"])
    if lite: style = "lite"
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
            if lite:                                                                                # far blocks: one band per floor
                obox(bm, (p[0] + ux * d / 2 + nx * deep / 2, p[1] + uy * d / 2 + ny * deep / 2), (ux, uy), d - 0.8, deep, z - 0.16, z + 0.95, white, PLAIN); continue
            for s0, s1 in spans:
                m = (s0 + s1) / 2; ln = s1 - s0; cx, cy = p[0] + ux * m, p[1] + uy * m
                obox(bm, (cx + nx * deep / 2, cy + ny * deep / 2), (ux, uy), ln, deep, z - 0.16, z, white, PLAIN)
                if style == "rail":
                    obox(bm, (cx + nx * (deep - 0.04), cy + ny * (deep - 0.04)), (ux, uy), ln, 0.05, z + 0.95, z + 1.02, (0.12, 0.12, 0.12), PLAIN)
                    obox(bm, (cx + nx * (deep - 0.04), cy + ny * (deep - 0.04)), (ux, uy), ln, 0.03, z + 0.1, z + 0.95, (0.30, 0.36, 0.38), GLASS)
                else:
                    obox(bm, (cx + nx * (deep - 0.05), cy + ny * (deep - 0.05)), (ux, uy), ln, 0.1, z, z + 0.95, white, PLAIN)
                    for e in (s0, s1): obox(bm, (p[0] + ux * e + nx * deep / 2, p[1] + uy * e + ny * deep / 2), (ux, uy), 0.1, deep, z, z + 0.95, white, PLAIN)
            for i in range(nb):                                                                     # air-conditioning units on some balconies
                if rnd.random() < 0.2:
                    a0 = 0.4 + (i + rnd.uniform(0.2, 0.8)) * bw; obox(bm, (p[0] + ux * a0 + nx * 0.22, p[1] + uy * a0 + ny * 0.22), (ux, uy), 0.8, 0.32, z, z + 0.58, (0.80, 0.80, 0.78), PLAIN)
            for i in range(nb):                                                                     # awnings over some bays
                if rnd.random() > pawn: continue
                f = fab if rnd.random() < 0.8 else FABRIC[rnd.randrange(len(FABRIC))]; a0, a1 = 0.4 + i * bw + 0.25, 0.4 + (i + 1) * bw - 0.25
                A = [(p[0] + ux * a0, p[1] + uy * a0, z + 2.75), (p[0] + ux * a1, p[1] + uy * a1, z + 2.75),
                     (p[0] + ux * a1 + nx * (deep + 0.15), p[1] + uy * a1 + ny * (deep + 0.15), z + 2.05), (p[0] + ux * a0 + nx * (deep + 0.15), p[1] + uy * a0 + ny * (deep + 0.15), z + 2.05)]
                bm.add(A + [(A[2][0], A[2][1], z + 1.85), (A[3][0], A[3][1], z + 1.85)], [(0, 1, 2), (0, 2, 3), (3, 2, 4), (3, 4, 5)], col=f, mat=PLAIN)
    rp = body.buffer(-1.6)                                                                          # roof: stair bulkhead + solar heaters
    if rp.geom_type == "Polygon" and rp.area > 20:
        q = rp.representative_point(); obox(bm, (q.x, q.y), (1, 0), 4.5, 3.5, zt, zt + 2.7, wall, PLAIN); bx = rp.bounds; a = rnd.uniform(0, 3.14)
        if lite: return
        obox(bm, (q.x + 1.5, q.y + 1.0), (1, 0), 0.06, 0.06, zt + 2.7, zt + 2.7 + rnd.uniform(2.5, 4.5), (0.15, 0.15, 0.15), PLAIN)          # TV aerial
        obox(bm, (q.x + 1.5, q.y + 1.0), (math.cos(a), math.sin(a)), 1.2, 0.04, zt + 4.6, zt + 4.64, (0.15, 0.15, 0.15), PLAIN)
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
    q = at(0, 2.03); LABELS.append({"text": "ΕΤΑΙΡΕΙΑ ΜΑΚΕΔΟΝΙΚΩΝ ΣΠΟΥΔΩΝ", "pos": [q[0], q[1], z1 + 0.45], "dir": [-n[1], n[0]], "n": [n[0], n[1]], "size": 1.05})
def theatre_royal(P, bd, zg, rnd):
    """Royal Theatre: long low hall in the park, shallow metal roof, a row of piers along each long side."""
    conc = (0.66, 0.64, 0.59); zt = zg + 9.5; walls(P, zg - 1.5, zt, zg, conc, WALLM, rnd)
    sides = mrr_sides(P); long = max(sides, key=lambda sd: sd[3]); short = min(sides, key=lambda sd: sd[3]); cen = P.centroid
    u = long[1]; n = long[2]; hl = long[3] / 2 + 1.5; hw = short[3] / 2 + 4.0; metal = (0.50, 0.52, 0.54)
    def pt(a, b, z): return (cen.x + u[0] * a + n[0] * b, cen.y + u[1] * a + n[1] * b, z)
    bm.add([pt(-hl, -hw, zt + 0.3), pt(hl, -hw, zt + 0.3), pt(hl, hw, zt + 0.3), pt(-hl, hw, zt + 0.3)], [(0, 1, 2), (0, 2, 3)], col=(0.60, 0.61, 0.62), mat=ROOFM)
    for sgn in (-1, 1):
        obox(bm, pt(0, sgn * hw, 0)[:2], u, 2 * hl, 0.4, zt - 0.5, zt + 0.9, conc, PLAIN)
        for i in range(int(2 * hl / 7)): obox(bm, pt(-hl + 5 + i * 7, sgn * hw * 0.45, 0)[:2], u, 2.6, 1.3, zt + 0.3, zt + 0.75, (0.04, 0.05, 0.07), GLASS)
    for sgn in (-1, 1): obox(bm, pt(sgn * hl, 0, 0)[:2], (n[0], n[1]), 2 * hw, 0.4, zt - 0.5, zt + 0.9, conc, PLAIN)
    bm.add([pt(-hl, -hw, zt - 0.5), pt(hl, -hw, zt - 0.5), pt(hl, hw, zt - 0.5), pt(-hl, hw, zt - 0.5)], [(0, 2, 1), (0, 3, 2)], col=conc, mat=PLAIN)
    for sgn in (-1, 1):
        obox(bm, pt(0, sgn * (hw - 0.3), 0)[:2], u, 2 * hl, 0.5, zt - 0.5, zt + 0.1, conc, PLAIN)
        for i in range(int(2 * hl / 5.5) + 1):
            a = -hl + 0.8 + i * (2 * hl - 1.6) / int(2 * hl / 5.5); obox(bm, pt(a, sgn * (hw - 0.6), 0)[:2], u, 0.7, 0.7, zg, zt - 0.5, conc, PLAIN)
OCHRE = (0.80, 0.58, 0.24); REDCOL = (0.42, 0.10, 0.08)
STY = {"144215746": dict(sign="ELECTRA PALACE HOTEL", cream=(0.84, 0.78, 0.63), cols=REDCOL, top="pergola", accent=None, awn=False),
       "26751274": dict(sign="ΟΛΥΜΠΙΟΝ", cream=(0.84, 0.77, 0.61), cols=REDCOL, top="tile", accent=None, awn=True)}
def hebrard(P, bd, zg, zt, rnd, lite, st):
    """Buildings on Aristotelous Square and its axis (Hebrard plan). Read off aerial and street photographs:
    a tall arcade of round arches with engaged red columns and a recessed gallery; above it rows of windows, each with its own
    balustraded balcony on brackets; one floor treated as a continuous loggia of small arches; a bracketed cornice; and on top
    either set-back terraces behind white pergolas or a red tiled roof. Heights are from the map; bay widths and mouldings are estimates."""
    cream = st["cream"]; trim = (0.92, 0.89, 0.82); dark = (0.03, 0.035, 0.045); dk = tuple(k * 0.72 for k in cream); iron = (0.08, 0.08, 0.08)
    H = zt - zg; ext = P.exterior if P.exterior.is_ccw else P.exterior.reverse(); c = list(ext.coords)[:-1]; E = []
    for i in range(len(c)):
        p_, r_ = c[i], c[(i + 1) % len(c)]; d = math.dist(p_, r_)
        if d < 0.05: continue
        u = ((r_[0] - p_[0]) / d, (r_[1] - p_[1]) / d); n = (u[1], -u[0]); mid = Point((p_[0] + r_[0]) / 2 + n[0] * 2.0, (p_[1] + r_[1]) / 2 + n[1] * 2.0)
        E.append((p_, r_, d, u, n, not any(allp[k].contains(mid) for k in alltree.query(mid))))
    G = 8.4 if H > 24 else 7.2; crown = 1.0; nfl = max(2, round((H - G - crown) / 3.3)); fh = (H - G - crown) / nfl; logk = nfl - 2 if nfl >= 4 else -1; topk = nfl - 1; pergola = st["top"] == "pergola" and nfl >= 3
    if st["top"] == "tile":
        Pe = P.buffer(0.55, join_style=2); inner = P.buffer(-3.4, join_style=2); tile = (0.50, 0.20, 0.11)
        if inner.geom_type == "Polygon" and inner.area > 30:
            ring = Pe.difference(inner); ib = inner.exterior
            for pp in (ring.geoms if hasattr(ring, "geoms") else [ring]):
                if pp.geom_type != "Polygon": continue
                v, t = tri(pp.segmentize(3.0)); z = np.array([zt + 2.3 if ib.distance(Point(q)) < 0.02 else zt - 0.05 for q in v]); bm.add(np.c_[v, z], t, col=tile, mat=PLAIN)
            roof(inner, zt + 2.3, tile, PLAIN)
            for k in range(int(inner.exterior.length / 9)):                              # dormers
                q = inner.exterior.interpolate(k * 9 + 3); q2 = inner.exterior.interpolate(k * 9 + 3.5); u = (q2.x - q.x, q2.y - q.y); nn = math.hypot(*u) or 1
                obox(bm, (q.x, q.y), (u[0] / nn, u[1] / nn), 1.3, 1.6, zt + 1.2, zt + 2.6, trim, PLAIN)
        else: roof(P, zt - 0.1, tile, PLAIN)
    elif pergola:
        inner = P.buffer(-2.3, join_style=2); roof(inner if inner.geom_type == "Polygon" and not inner.is_empty else P, zt - 0.9, (0.58, 0.57, 0.55))
    else:
        roof(P, zt - 0.9, (0.58, 0.57, 0.55)); inner = P.buffer(-3.5, join_style=2)
        if inner.geom_type == "Polygon" and inner.area > 0.35 * P.area: walls(inner, zt - 0.9, zt + 2.4, zg, cream, WALLM, rnd); roof(inner, zt + 2.4, (0.6, 0.59, 0.57))
    def plain(e):
        p_, r_, d = e[0], e[1], e[2]; bm.add([(p_[0], p_[1], zg - 1.5), (r_[0], r_[1], zg - 1.5), (r_[0], r_[1], zt), (p_[0], p_[1], zt)], Q, uv=[(0, -1.5), (d, -1.5), (d, H), (0, H)], col=cream, mat=WALLM)
    def facade(path, sign):
        cum = [0.0]
        for e in path: cum.append(cum[-1] + e[2])
        Ls = cum[-1]
        def seg(sv):
            j = 0
            while j < len(path) - 1 and sv > cum[j + 1]: j += 1
            return j
        def P3(sv, z, dep=0.0):
            sv = min(max(sv, 0.0), Ls); j = seg(sv); p_, _, _, u, n, _ = path[j]; a = sv - cum[j]; return (p_[0] + u[0] * a - n[0] * dep, p_[1] + u[1] * a - n[1] * dep, zg + z)
        def cellmesh(poly, dep, col, mat=PLAIN):
            for pp in (poly.geoms if hasattr(poly, "geoms") else [poly]):
                if pp.geom_type == "Polygon" and pp.area > 1e-4: v, t = tri(pp); bm.add([P3(a, b, dep) for a, b in v], t, col=col, mat=mat)
        def reveal(poly, d0, d1, col, skip_bottom=False):
            cc = list(poly.exterior.coords); zmin = min(b for _, b in cc)
            for (a, b), (e_, f) in zip(cc[:-1], cc[1:]):
                if skip_bottom and abs(b - zmin) < 1e-6 and abs(f - zmin) < 1e-6: continue
                bm.add([P3(a, b, d0), P3(e_, f, d0), P3(e_, f, d1), P3(a, b, d1)], Q, col=col, mat=PLAIN)
        def band(s0, s1, z0, z1, out, col, mat=PLAIN, inner=0.0):
            s0, s1 = max(s0, 0.0), min(s1, Ls)
            if s1 - s0 < 0.01: return
            A = [P3(s0, z0, -out), P3(s1, z0, -out), P3(s1, z1, -out), P3(s0, z1, -out), P3(s0, z0, -inner), P3(s1, z0, -inner), P3(s1, z1, -inner), P3(s0, z1, -inner)]
            bm.add(A, [(0, 1, 2), (0, 2, 3), (3, 2, 6), (3, 6, 7), (0, 5, 1), (0, 4, 5), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2), (4, 6, 5), (4, 7, 6)], col=col, mat=mat)
        def quad(s0, s1, z0, z1, dep, col, mat=PLAIN): bm.add([P3(s0, z0, dep), P3(s1, z0, dep), P3(s1, z1, dep), P3(s0, z1, dep)], Q, col=col, mat=mat)
        def hquad(s0, s1, z, d0, d1, col): bm.add([P3(s0, z, d0), P3(s1, z, d0), P3(s1, z, d1), P3(s0, z, d1)], Q, col=col, mat=PLAIN)
        def balustrade(sL, sR, zf, out, inn):
            """rail on turned balusters between end posts, standing on the edge of a slab."""
            if lite: band(sL, sR, zf, zf + 0.95, out, trim, inner=inn); return
            band(sL, sR, zf, zf + 0.09, out, trim, inner=inn); band(sL, sR, zf + 0.86, zf + 0.96, out + 0.02, trim, inner=inn - 0.02)
            for e_ in (sL, sR - 0.14): band(e_, e_ + 0.14, zf + 0.09, zf + 0.86, out, trim, inner=inn)
            nbal = max(2, int((sR - sL - 0.3) / 0.2))
            for q_ in range(nbal): a = sL + 0.22 + q_ * (sR - sL - 0.44) / max(1, nbal - 1); band(a - 0.035, a + 0.035, zf + 0.09, zf + 0.86, out - 0.03, trim, inner=inn + 0.03)
        nb = max(2, round(Ls / 4.3)); bw = Ls / nb; acc = st["accent"]
        for i in range(nb):
            s0, s1 = i * bw, (i + 1) * bw; sm = (s0 + s1) / 2; hw = (bw - 1.3) / 2; top = G - 1.55
            # ---- arcade ----
            arch = Polygon([(sm - hw, 0), (sm + hw, 0)] + [(sm + hw * math.cos(math.pi * k / 14), top - hw + hw * math.sin(math.pi * k / 14)) for k in range(15)])
            cellmesh(box(s0, 0, s1, G).difference(arch), 0, cream); reveal(arch, 0, 0.9, cream, True)
            quad(s0, s1, 0, G, 3.6, dk); hquad(s0, s1, G - 0.3, 0.9, 3.6, dk)
            quad(s0 + 0.35, s1 - 0.35, 0.45, 3.6, 3.54, dark, GLASS); quad(s0 + 0.6, s1 - 0.6, 4.6, 6.2, 3.54, dark, GLASS)
            if rnd.random() < 0.6: quad(s0 + 0.35, s1 - 0.35, 3.6, 4.25, 3.50, FABRIC[rnd.randrange(len(FABRIC))])
            if not lite:
                arc = [(sm + (hw + 0.16) * math.cos(math.pi * k / 14), top - hw + (hw + 0.16) * math.sin(math.pi * k / 14)) for k in range(15)]      # moulded archivolt
                arc_i = [(sm + (hw + 0.0) * math.cos(math.pi * k / 14), top - hw + hw * math.sin(math.pi * k / 14)) for k in range(15)]
                for (a0, b0), (a1, b1), (c0, d0), (c1, d1) in zip(arc[:-1], arc[1:], arc_i[:-1], arc_i[1:]): bm.add([P3(c0, d0, -0.07), P3(c1, d1, -0.07), P3(a1, b1, -0.07), P3(a0, b0, -0.07)], Q, col=trim, mat=PLAIN)
                band(sm - 0.16, sm + 0.16, top - 0.05, top + 0.42, 0.14, trim)                                                           # keystone
                colr = st["cols"]; cz = top - hw                                                                                           # engaged column on the pier
                band(s0 - 0.42, s0 + 0.42, 0, 0.75, 0.5, trim); limb(bm, P3(s0, 0.75, -0.16), P3(s0, cz - 0.35, -0.16), 0.33, 0.29, colr, PLAIN, 12, False)
                band(s0 - 0.40, s0 + 0.40, cz - 0.35, cz - 0.12, 0.5, trim); band(s0 - 0.46, s0 + 0.46, cz - 0.12, cz, 0.56, trim)
            else: band(s0 - 0.33, s0 + 0.33, 0, top - hw, 0.3, st["cols"])
            # ---- floors ----
            ww = min(1.6, bw * 0.4)
            for k in range(nfl):
                z0 = G + k * fh
                if k == logk:                                                                                                              # loggia of small arches
                    wl = bw / 2 - 0.55; ops = []
                    for cs in (sm - bw / 4, sm + bw / 4):
                        tp = z0 + fh - 0.5; ops.append(Polygon([(cs - wl / 2, z0), (cs + wl / 2, z0)] + [(cs + wl / 2 * math.cos(math.pi * q_ / 10), tp - wl / 2 + wl / 2 * math.sin(math.pi * q_ / 10)) for q_ in range(11)]))
                    cellmesh(box(s0, z0, s1, z0 + fh).difference(unary_union(ops)), 0, acc or cream)
                    for o in ops: reveal(o, 0, 0.35, trim, True); ob = o.bounds; balustrade(ob[0], ob[2], z0, -0.06, -0.22)
                    quad(s0, s1, z0, z0 + fh, 1.5, dk); hquad(s0, s1, z0 + 0.01, 0, 1.5, dk); hquad(s0, s1, z0 + fh - 0.08, 0.35, 1.5, dk); quad(sm - 0.75, sm + 0.75, z0 + 0.05, z0 + 2.3, 1.46, dark, GLASS)
                    if not lite:
                        for cs in (sm - bw / 4, sm + bw / 4):
                            for e_ in (-wl / 2, wl / 2): band(cs + e_ - 0.12, cs + e_ + 0.12, z0 + fh - 0.5 - wl / 2 - 0.16, z0 + fh - 0.5 - wl / 2, 0.08, trim)   # imposts
                    continue
                if k == topk and pergola:                                                                                                  # set-back terrace behind a white pergola
                    quad(s0, s1, z0, z0 + 3.1, 2.3, cream); hquad(s0, s1, z0 + 0.01, 0, 2.3, dk); quad(sm - 0.8, sm + 0.8, z0 + 0.05, z0 + 2.35, 2.26, dark, GLASS)
                    balustrade(s0, s1, z0, -0.02, -0.2); band(s0 - 0.09, s0 + 0.09, z0, z0 + 2.9, -0.22, trim, inner=-0.4); band(s0, s1, z0 + 2.9, z0 + 3.12, -0.18, trim, inner=-0.44)
                    for q_ in range(4): a = s0 + (q_ + 0.5) * bw / 4; band(a - 0.05, a + 0.05, z0 + 3.12, z0 + 3.28, -0.1, trim, inner=-2.3)
                    if rnd.random() < 0.4: hquad(s0 + 0.1, s1 - 0.1, z0 + 3.3, 0.3, 2.2, (0.90, 0.88, 0.80))                                    # canvas awning
                    continue
                arched = k == topk and st["top"] == "tile"
                if arched:
                    wt = z0 + fh - 0.7; op = Polygon([(sm - ww / 2, z0 + 0.3), (sm + ww / 2, z0 + 0.3)] + [(sm + ww / 2 * math.cos(math.pi * q_ / 10), wt - ww / 2 + ww / 2 * math.sin(math.pi * q_ / 10)) for q_ in range(11)])
                else: op = box(sm - ww / 2, z0 + 0.3, sm + ww / 2, z0 + fh - 0.75)
                cellmesh(box(s0, z0, s1, z0 + fh).difference(op), 0, cream); reveal(op, 0, 0.28, trim); cellmesh(op, 0.28, dark, GLASS); ob = op.bounds
                if not lite:
                    band(sm - 0.035, sm + 0.035, ob[1], ob[3] - (ww / 2 if arched else 0), -0.26, trim, inner=-0.2); band(ob[0], ob[2], ob[1] + (ob[3] - ob[1]) * 0.68, ob[1] + (ob[3] - ob[1]) * 0.68 + 0.06, -0.26, trim, inner=-0.2)
                    for e_ in (ob[0] - 0.13, ob[2]): band(e_, e_ + 0.13, ob[1], ob[3] - (ww / 2 if arched else 0), 0.06, trim)                    # architrave
                if not arched: band(ob[0] - 0.2, ob[2] + 0.2, ob[3], ob[3] + 0.16, 0.08, trim); band(ob[0] - 0.26, ob[2] + 0.26, ob[3] + 0.16, ob[3] + 0.26, 0.2, trim)   # lintel with a small cornice
                if st["awn"] and arched and rnd.random() < 0.6:
                    fc = (0.10, 0.38, 0.28) if rnd.random() < 0.6 else (0.55, 0.10, 0.08); bm.add([P3(ob[0] - 0.1, ob[3], 0), P3(ob[2] + 0.1, ob[3], 0), P3(ob[2] + 0.1, ob[3] - ww / 2 - 0.2, -0.9), P3(ob[0] - 0.1, ob[3] - ww / 2 - 0.2, -0.9)], Q, col=fc, mat=PLAIN)
                # balcony on brackets with a balustrade
                sL, sR = ob[0] - 0.45, ob[2] + 0.45; band(sL, sR, z0 + 0.14, z0 + 0.3, 0.78, trim)
                if not lite:
                    for e_ in (sL + 0.12, sR - 0.3): band(e_, e_ + 0.18, z0 - 0.22, z0 + 0.14, 0.5, trim); band(e_, e_ + 0.18, z0 - 0.42, z0 - 0.22, 0.26, trim)
                balustrade(sL, sR, z0 + 0.3, 0.78, 0.66)
                if not lite:
                    for e_ in (sL, sR - 0.12): band(e_, e_ + 0.12, z0 + 0.3, z0 + 1.26, 0.66, trim)
            ptop = G + (logk if logk >= 0 else (topk if pergola else nfl)) * fh
            band(s0 - 0.3, s0 + 0.3, G + 0.3, ptop - 0.2, 0.14, trim); band(s0 - 0.36, s0 + 0.36, ptop - 0.5, ptop - 0.2, 0.2, trim)                    # pilaster and its capital
            band(s0, s1, G - 0.2, G + 0.12, 0.34, trim); band(s0, s1, G + 0.12, G + 0.3, 0.5, trim)                                                  # cornice over the arcade
            zc = G + (topk if pergola else nfl) * fh                                                                                                  # main cornice
            if logk >= 0: band(s0, s1, G + logk * fh - 0.2, G + logk * fh + 0.02, 0.3, trim); band(s0, s1, G + (logk + 1) * fh - 0.14, G + (logk + 1) * fh + 0.06, 0.22, trim)
            band(s0, s1, zc - 0.2, zc + 0.04, 0.42, trim); band(s0, s1, zc + 0.04, zc + 0.24, 0.66, trim)
            if not lite:
                for q_ in range(int(bw / 0.62)): a = s0 + 0.2 + q_ * 0.62; band(a - 0.09, a + 0.09, zc - 0.42, zc - 0.2, 0.4, trim)                   # modillions
                for q_ in range(int(bw / 0.3)): a = s0 + 0.1 + q_ * 0.3; band(a - 0.06, a + 0.06, G - 0.36, G - 0.2, 0.2, trim)                       # dentils
            if not pergola: cellmesh(box(s0, zc, s1, H), 0, cream); band(s0, s1, H - 0.14, H, 0.08, trim)
        band(Ls - 0.3, Ls, G + 0.3, G + (logk if logk >= 0 else nfl) * fh - 0.2, 0.14, trim)
        if sign and not lite:
            adv = [0.45 if ch == " " else (0.36 if ch in "IΙ" else 0.86) for ch in sign]; tot = sum(adv) * 0.8; sc = Ls / 2 - tot / 2
            for ch, ad in zip(sign, adv):
                if ch != " ":
                    j = seg(sc + ad * 0.4); q = P3(sc + ad * 0.4, G - 1.28, -0.04); LABELS.append({"text": ch, "pos": [q[0], q[1], q[2]], "dir": list(path[j][3]), "n": list(path[j][4]), "size": 0.9})
                sc += ad * 0.8
    start = next((i for i, e in enumerate(E) if not e[5]), 0); cur = []; paths = []
    def flush():
        if not cur: return
        if sum(e[2] for e in cur) >= 7: paths.append(list(cur))
        else:
            for e in cur: plain(e)
        cur.clear()
    for k in range(len(E)):
        e = E[(start + k) % len(E)]
        if e[5] and (not cur or cur[-1][3][0] * e[3][0] + cur[-1][3][1] * e[3][1] > 0.82): cur.append(e)
        else:
            flush()
            if e[5]: cur.append(e)
            else: plain(e)
    flush(); hub = Point(Wd(70, 0))
    def pmid(pth): k = len(pth) // 2; return Point((pth[k][0][0] + pth[k][1][0]) / 2, (pth[k][0][1] + pth[k][1][1]) / 2)
    best = min(paths, key=lambda pth: pmid(pth).distance(hub) - 0.2 * sum(e[2] for e in pth)) if paths and st.get("sign") else None
    for pth in paths: facade(pth, st.get("sign") if pth is best else None)
HB = Mesh(); HW = Mesh()
NHEB = 0
for bd, P in zip(L["buildings"], allp):
    if bd["id"] == TOWER_ID:
        c = P.centroid; tower = {"x": c.x, "y": c.y, "r_osm": math.sqrt(P.area / math.pi), "ground": land_h(c.x, c.y)}; continue
    if P.is_empty or P.geom_type != "Polygon": continue
    polys.append(P); rnd = random.Random(bd["id"]); dist = P.centroid.distance(TW)
    if bd["src"] == "estimate" and P.area < 110 and dist < 300: bd["h"] = 3.2 if P.area < 70 else 6.0   # kiosks and sheds, not flats
    near = dist < INNER or P.centroid.distance(AC) < 300
    wall = (NEARPAL if near else PAL)[rnd.randrange(len(NEARPAL if near else PAL))]; k = rnd.uniform(0.9, 1.05); wall = tuple(min(1, c * k) for c in wall)
    small = P.area < 45 or bd["h"] < 5
    if bd["kind"] in ("church", "cathedral", "chapel") or bd.get("hist"): wall = (0.72, 0.56, 0.44)
    hs = [land_h(*p) for p in P.exterior.coords]; zb = min(hs) - 1.5; zg = sum(hs) / len(hs); zt = zg + bd["h"]
    if bd["id"] == THEATRE_KTHBE: theatre_kthbe(P, bd, zg, rnd); continue
    if P.geom_type == "Polygon" and P.area > 200 and bd["h"] > 12 and P.centroid.distance(AC) < 330 and P.distance(APLZ) < 5:
        st = STY.get(bd["id"]) or dict(sign=None, cream=[(0.82, 0.75, 0.60), (0.86, 0.80, 0.66), (0.80, 0.73, 0.60), (0.88, 0.84, 0.74)][rnd.randrange(4)], cols=REDCOL if rnd.random() < 0.4 else (0.86, 0.82, 0.72),
                                       top=rnd.choice(["pergola", "pergola", "tile", "flat"]), accent=OCHRE if rnd.random() < 0.45 else None, awn=rnd.random() < 0.3)
        BM0 = bm
        for tgt, lite in ((HB, False), (HW, True)): bm = tgt; hebrard(P, bd, zg, zt, random.Random(bd["id"]), lite, st)
        bm = BM0; NHEB += 1; continue
    if bd["id"] == THEATRE_ROYAL: theatre_royal(P, bd, zg, rnd); continue
    if bd["id"] == OFFICERS:                                                    # tiled hipped roof
        wall = (0.86, 0.80, 0.66); walls(P, zb, zt, zg, wall, WALLM, rnd); c = list(P.exterior.coords); q = P.centroid
        for p, r in zip(c[:-1], c[1:]): bm.add([(p[0], p[1], zt), (r[0], r[1], zt), (q.x, q.y, zt + 4.5)], [(0, 1, 2)], col=(0.52, 0.20, 0.10), mat=PLAIN)
        continue
    if (near or dist < 650) and bd["h"] > 11 and not small and bd["kind"] not in ("church", "school", "sports_centre", "theatre", "military") and not bd["min_h"]:
        polykatoikia(P, bd, zg, zt, wall, rnd, lite=not near); continue
    if bd["min_h"]: zb = zg + bd["min_h"]
    walls(P, zb, zt, zg, wall, PLAIN if small else WALLM, rnd); roof(P, zt, ROOF[rnd.randrange(len(ROOF))])
bm.save("buildings"); HB.save("aris_bld"); HW.save("aris_bld_web"); print("hebrard buildings", NHEB); json.dump(LABELS, open(f"{root}/data/labels.json", "w"), ensure_ascii=False)
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
_own = unary_union([Polygon(g_.exterior.coords) for g_ in (APLZ.geoms if hasattr(APLZ, 'geoms') else [APLZ])] + [Point(Wd(70, 0)).buffer(34)]).buffer(3); _pown = prep(_own)
spots = [p_ for p_ in spots if not _pown.contains(Point(p_))]          # the square is planted by hand further down
tm = Mesh()
for (x, y) in spots:
    z = land_h(x, y); hgt = rnd.uniform(6, 11); r = hgt * rnd.uniform(0.26, 0.34)
    gcol = (rnd.uniform(0.07, 0.13), rnd.uniform(0.15, 0.24), rnd.uniform(0.04, 0.08))
    tr = [(x + 0.18 * math.cos(k * 2.094), y + 0.18 * math.sin(k * 2.094), zz) for zz in (z, z + hgt * 0.45) for k in range(3)]
    tm.add(tr, [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)], col=(0.16, 0.11, 0.07), mat=0)
    WEB_TREES.append([round(x, 2), round(y, 2), round(z, 2), round(hgt, 2), round(r, 2)] + [round(c, 3) for c in gcol])
    close = math.hypot(x - TX, y - TY) < 520 or math.hypot(x - ACX, y - ACY) < 330; sg, rg = (10, 6) if close else (6, 4)
    for k in range(4 if close else 2):
        off = 0 if k == 0 else 0.55; kk = 1 if k == 0 else rnd.uniform(0.55, 0.8); gc = tuple(c * rnd.uniform(0.85, 1.15) for c in gcol)
        bx, by, bz = x + rnd.uniform(-1, 1) * r * off, y + rnd.uniform(-1, 1) * r * off, z + hgt * (0.68 if k == 0 else rnd.uniform(0.5, 0.75))
        sh = 0.88 if close else 1.0
        v, t = blob(bx, by, bz, r * kk * sh, hgt * 0.34 * kk * sh, rnd, sg, rg); tm.add(v, t, col=tuple(c * 0.55 for c in gc) if close else gc, mat=0)
        if close: foliage(tm, bx, by, bz, r * kk, hgt * 0.34 * kk, int(420 * LEAF), 0.32 / LEAF ** 0.5, gc, rnd)
tm.save("trees"); print("trees", len(spots))

# ================= the tower's surroundings (within 260 m) =================
RS = 260.0; disc = Point(TX, TY).buffer(RS, 64); land = disc.difference(sea); S = Mesh()
GRASS, PAVE, COBBLE, ASPHALT, CURB, PROM, LAMP, GLOW, CAR, CARGLASS, BARK, PINE, MARK, WALK, TERRA, MARBLE, DIRT, STELE, GREY, MATTE = range(20)
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
SP0 = np.array([TX + 47, TY - 27]); SP1 = np.array([TX - 15, TY - 113]); _s = (SP1 - SP0) / np.linalg.norm(SP1 - SP0); _w = np.array([_s[1], -_s[0]])   # the stelae line and the direction of the walk beside it
strip = Polygon([SP0 - _w * 3, SP1 - _w * 3, SP1 + _w * 34, SP0 + _w * 34]).intersection(land).difference(prom).difference(drive.buffer(2.6)); terra = terra.difference(strip)
flat(land.difference(bu.buffer(0.2)), 0.02, PAVE)
flat(terra, 0.03, TERRA)
flat(strip, 0.035, GREY)
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
def car(c, ux, uy, z=None, wheels=True):
    col = CARCOL[crnd.randrange(len(CARCOL))]; ln = crnd.uniform(3.9, 4.6)
    body = [(-.5, .30), (-.5, .62), (-.46, .76), (-.22, .82), (.20, .84), (.46, .72), (.5, .52), (.5, .30)]
    cab = [(-.37, .80), (-.27, 1.30), (.08, 1.33), (.25, .83)]; van = crnd.random() < 0.1
    if van: body = [(-.5, .30), (-.5, 1.75), (.30, 1.78), (.42, 1.25), (.5, .8), (.5, .30)]; cab = [(.26, 1.2), (.31, 1.7), (.33, 1.7), (.44, 1.2)]; col = (0.82, 0.82, 0.8)
    Z = FLAT + 0.06 if z is None else z
    prism(S, [(a * ln, h) for a, h in body], c, (ux, uy), 0.88, Z, col, CAR)
    prism(S, [(a * ln, h) for a, h in cab], c, (ux, uy), 0.80 if not van else 0.86, Z, (1, 1, 1), CARGLASS)
    if not van: prism(S, [(-.27 * ln, 1.29), (-.27 * ln, 1.34), (.08 * ln, 1.37), (.08 * ln, 1.32)], c, (ux, uy), 0.81, Z, col, CAR)
    if not wheels: return
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
# ---- life in the square: people, bins, park lights, traffic lights, tour boats ----
lrnd = random.Random(8)
SKIN = [(0.78, 0.60, 0.48), (0.70, 0.52, 0.40), (0.85, 0.68, 0.56), (0.55, 0.38, 0.28)]
SHIRT = [(0.85, 0.85, 0.83), (0.10, 0.10, 0.12), (0.65, 0.12, 0.10), (0.12, 0.25, 0.50), (0.80, 0.70, 0.20), (0.20, 0.42, 0.30), (0.55, 0.55, 0.58), (0.86, 0.50, 0.20), (0.45, 0.25, 0.45)]
PANTS = [(0.10, 0.12, 0.20), (0.08, 0.08, 0.09), (0.30, 0.32, 0.38), (0.45, 0.38, 0.28), (0.15, 0.20, 0.32)]
def person(x, y, a, k=1.0, S=None):
    tgt = S if S is not None else globals()["S"]; person2(tgt, x, y, FLAT + 0.04, a, k, lrnd, MATTE if S is None else AMATTE)
walkable = unary_union([prom, terra, strip]).difference(grass.buffer(0.7)).difference(ringpoly.buffer(0.6)).difference(lawn.buffer(0.3)).difference(drive.buffer(3)); pwalk = prep(walkable); wb = walkable.bounds; npeople = 0
while npeople < 190:
    x, y = lrnd.uniform(wb[0], wb[2]), lrnd.uniform(wb[1], wb[3])
    if not pwalk.contains(Point(x, y)) or math.hypot(x - TX, y - TY) > 200: continue
    a = lrnd.uniform(0, 6.28); k = 0.62 if lrnd.random() < 0.08 else lrnd.uniform(0.93, 1.07); person(x, y, a, k); npeople += 1
    if lrnd.random() < 0.4 and pwalk.contains(Point(x - math.sin(a) * 0.62, y + math.cos(a) * 0.62)): person(x - math.sin(a) * 0.62, y + math.cos(a) * 0.62, a + lrnd.uniform(-0.2, 0.2), lrnd.uniform(0.9, 1.05)); npeople += 1
for i in range(int(LN / 11.5)):                                                                   # litter bins by the benches
    m = 6 + i * 11.5 + 3.3; q = (P0[0] + sx * m + wx * 4.3, P0[1] + sy * m + wy * 4.3)
    if pland.contains(Point(q)): obox(S, q, (sx, sy), 0.45, 0.45, FLAT + 0.02, FLAT + 0.9, (1, 1, 1), LAMP)
for off in (10.5, 22.0):                                                                          # tall grey park lights with twin heads
    for i in range(int(LN / 24) + 1):
        m = 8 + i * 24; q = (P0[0] + sx * m + wx * off, P0[1] + sy * m + wy * off)
        if not pland.contains(Point(q)): continue
        obox(S, q, (sx, sy), 0.16, 0.16, FLAT, FLAT + 8.0, (0.5, 0.5, 0.52), MATTE); obox(S, q, (sx, sy), 1.5, 0.08, FLAT + 7.9, FLAT + 8.0, (0.5, 0.5, 0.52), MATTE)
        for e in (-0.75, 0.75): obox(S, (q[0] + sx * e, q[1] + sy * e), (sx, sy), 0.5, 0.32, FLAT + 7.75, FLAT + 7.9, (1, 1, 1), GLOW)
nsig = 0; seen = set()
for jx, jy in JUNC:                                                                               # traffic lights at the junctions
    key = (round(jx / 3), round(jy / 3))
    if key in seen or math.hypot(jx - TX, jy - TY) > RS - 20: continue
    seen.add(key)
    for a in (0.8, 2.4, 3.9, 5.5):
        for rr in (7, 10, 13):
            q = Point(jx + rr * math.cos(a), jy + rr * math.sin(a))
            if walk.contains(q):
                obox(S, (q.x, q.y), (1, 0), 0.12, 0.12, FLAT + 0.2, FLAT + 3.3, (1, 1, 1), LAMP); obox(S, (q.x, q.y), (math.cos(a), math.sin(a)), 0.3, 0.3, FLAT + 2.5, FLAT + 3.45, (0.03, 0.03, 0.03), MATTE)
                for kz, cc in ((3.25, (0.9, 0.05, 0.05)), (2.95, (0.9, 0.6, 0.05)), (2.65, (0.05, 0.7, 0.2))): obox(S, (q.x - math.cos(a) * 0.16, q.y - math.sin(a) * 0.16), (math.cos(a), math.sin(a)), 0.04, 0.18, FLAT + kz - 0.09, FLAT + kz + 0.09, cc, MATTE)
                nsig += 1; break
def hull(cx, cy, ang, Lb, Wb, z0, z1, col, deck):
    n = 14; half = []
    for i in range(n + 1):
        sp = -Lb / 2 + Lb * i / n; t = sp / (Lb / 2); w = Wb / 2 * (1 - max(0, t) ** 2.4) * (1 - 0.35 * max(0, -t) ** 3); half.append((sp, max(w, 0.05)))
    ring = [(a, w) for a, w in half] + [(a, -w) for a, w in half[::-1]]; ca, sa = math.cos(ang), math.sin(ang)
    pts = [(cx + a * ca - w * sa, cy + a * sa + w * ca) for a, w in ring]; Pg = Polygon(pts).buffer(0)
    c = list(Pg.exterior.coords)
    for p_, r_ in zip(c[:-1], c[1:]): S.add([(p_[0], p_[1], z0), (r_[0], r_[1], z0), (r_[0], r_[1], z1), (p_[0], p_[1], z1)], Q, col=col, mat=MATTE); S.add([(r_[0], r_[1], z0), (p_[0], p_[1], z0), (p_[0], p_[1], z1), (r_[0], r_[1], z1)], Q, col=col, mat=MATTE)
    v, t = tri(Pg); S.add(np.c_[v, np.full(len(v), z1 - 0.3)], t, col=deck, mat=MATTE); return (ca, sa)
for cx, cy, ang in ((TX - 66, TY - 56, 2.2), (TX - 59, TY - 90, 2.05)):                           # the wooden tour boats moored off the quay
    u = hull(cx, cy, ang, 22, 5.6, 0.2, 2.6, (0.20, 0.11, 0.06), (0.42, 0.28, 0.16))
    obox(S, (cx - u[0] * 6.5, cy - u[1] * 6.5), u, 6, 4.2, 2.3, 4.6, (0.26, 0.15, 0.08), MATTE); obox(S, (cx + u[0] * 7.5, cy + u[1] * 7.5), u, 3.5, 3.0, 2.3, 3.6, (0.26, 0.15, 0.08), MATTE)
    for m_, hgt in ((-1.5, 14.0), (4.5, 11.5)):
        q = (cx + u[0] * m_, cy + u[1] * m_); obox(S, q, u, 0.26, 0.26, 2.3, 2.3 + hgt, (0.25, 0.16, 0.09), MATTE)
        for hz in (0.55, 0.85): obox(S, q, (-u[1], u[0]), 5.5 * (1.3 - hz), 0.14, 2.3 + hgt * hz, 2.3 + hgt * hz + 0.14, (0.25, 0.16, 0.09), MATTE)
for cx, cy, ang, Lb in ((-1350, -950, 0.5, 150), (-2300, -1700, 1.2, 190), (-800, -2100, 2.6, 120)):   # ships in the gulf
    u = hull(cx, cy, ang, Lb, Lb * 0.16, 0.2, 9.0, (0.10, 0.12, 0.16), (0.45, 0.20, 0.16))
    obox(S, (cx - u[0] * Lb * 0.36, cy - u[1] * Lb * 0.36), u, Lb * 0.12, Lb * 0.13, 8.7, 25.0, (0.85, 0.85, 0.83), MATTE); obox(S, (cx - u[0] * Lb * 0.36, cy - u[1] * Lb * 0.36), u, 4, 4, 25, 31, (0.7, 0.3, 0.1), MATTE)
# ---- Mount Olympus across the gulf: a hazy ridge on the horizon (true bearing ~216 deg, drawn nearer and lower at the same angle) ----
mr = random.Random(2); D = 27000.0; prev = None
for az in range(184, 256):
    hgt = 1000 * math.exp(-((az - 216) / 8.0) ** 2) + 380 * math.exp(-((az - 236) / 7.0) ** 2) + 260 * math.exp(-((az - 199) / 9.0) ** 2) + 120 + mr.uniform(-45, 45)
    a = math.radians(az); cur = (D * math.sin(a), D * math.cos(a), hgt)
    if prev: S.add([(prev[0], prev[1], 0), (cur[0], cur[1], 0), (cur[0], cur[1], cur[2]), (prev[0], prev[1], prev[2])], [(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)], col=(0.36, 0.44, 0.54), mat=MATTE)
    prev = cur
# ---- cars in the rest of the city: sparse traffic and parked cars on every street outside the square ----
nfar = 0; frnd = crnd
def far_lane(ls, off, forward, lo, hi, skip):
    global nfar
    d = frnd.uniform(2, hi)
    while d < ls.length - 2.5:
        a, b = ls.interpolate(d - 0.5), ls.interpolate(d + 0.5); ux, uy = b.x - a.x, b.y - a.y; n = math.hypot(ux, uy) or 1; ux, uy = ux / n, uy / n
        q = ls.interpolate(d); c = (q.x - uy * off, q.y + ux * off); d += frnd.uniform(lo, hi)
        if frnd.random() < skip or math.hypot(c[0] - TX, c[1] - TY) < RS + 4: continue
        pt = Point(c)
        if psea.contains(pt) or any(polys[k].distance(pt) < 1.3 for k in btree.query(pt.buffer(1.5))): continue
        nfar += 1; car(c, ux, uy, land_h(*c) + 0.05, False) if forward else car(c, -ux, -uy, land_h(*c) + 0.05, False)
for r in L["roads"]:
    if r["kind"] not in DRIVE: continue
    ls = LineString(r["line"])
    if ls.length < 14: continue
    w = r["w"]; one = r.get("oneway")
    if w <= 8.5:
        for o in (-(w / 2 - 1.05), w / 2 - 1.05): far_lane(ls, o, one or o < 0, 5.4, 8.0, 0.45)
        far_lane(ls, 0.0 if one else -0.9, True, 40, 110, 0.3)
    else:
        nl_ = max(2, round(w / 3.4))
        for k in range(nl_): o = -w / 2 + (k + 0.5) * w / nl_; far_lane(ls, o, one or o < 0, 16, 55, 0.2)
# ---- stone pines: tall bare leaning trunks, broad ragged umbrella crowns; the largest stand close against the tower ----
prnd = random.Random(5); pines = []; forced = []
for deg, rr in ((128, 19.0), (152, 16.5), (176, 18.5), (204, 17.0), (232, 19.5), (258, 16.8), (287, 18.8), (316, 17.4), (300, 26.0), (215, 27.5), (160, 27.0)):
    q = rp(rr, math.radians(deg))
    if not doorway.buffer(1.5).contains(Point(q)) and not ringpoly.contains(Point(q)): forced.append(q); pines.append(q)
for (x, y) in sorted(near_spots, key=lambda p: prnd.random()):
    if all(math.hypot(x - a, y - b) > 5.2 for a, b in pines) and not drive.buffer(1).contains(Point(x, y)) and not ringpoly.buffer(0.8).contains(Point(x, y)) and not doorway.contains(Point(x, y)) and not strip.contains(Point(x, y)): pines.append((x, y))
for (x, y) in pines:
    dT = math.hypot(x - TX, y - TY); big = (x, y) in forced
    H = prnd.uniform(11.5, 14.5) if big else (prnd.uniform(8.5, 12.5) if dT < 75 else prnd.uniform(6.5, 10.5)); cr = H * prnd.uniform(0.42, 0.60)
    lean = math.atan2(y - TY, x - TX) + prnd.uniform(-0.9, 0.9) if dT < 35 else prnd.uniform(0, 2 * math.pi); lk = prnd.uniform(0.08, 0.24)
    lx, ly = math.cos(lean) * H * lk, math.sin(lean) * H * lk; K = 6; rings = []
    for i in range(K + 1):
        f = i / K; cx, cy, cz = x + lx * f ** 1.6, y + ly * f ** 1.6, FLAT + H * 0.74 * f; rr_ = 0.30 * (1 - 0.5 * f) * (H / 9)
        rings.append([(cx + rr_ * math.cos(j * math.pi / 3), cy + rr_ * math.sin(j * math.pi / 3), cz) for j in range(6)])
    v = [q for ring in rings for q in ring]; t = []
    for i in range(K):
        for j in range(6): a = i * 6 + j; b = i * 6 + (j + 1) % 6; t += [(a, b, b + 6), (a, b + 6, a + 6)]
    S.add(v, t, col=(1, 1, 1), mat=BARK); tx_, ty_, tz_ = x + lx, y + ly, FLAT + H * 0.74
    g = (prnd.uniform(0.030, 0.055), prnd.uniform(0.085, 0.135), prnd.uniform(0.022, 0.042))
    for k in range(prnd.randint(8, 12)):
        a = prnd.uniform(0, 2 * math.pi); d = cr * prnd.uniform(0.15, 0.9) if k else 0; rx = cr * prnd.uniform(0.32, 0.55)
        bx, by, bz = tx_ + d * math.cos(a), ty_ + d * math.sin(a), tz_ + H * prnd.uniform(0.04, 0.2) - d * 0.12
        rzz = rx * prnd.uniform(0.36, 0.52); gk = tuple(c * prnd.uniform(0.75, 1.3) for c in g)
        vv, tt = blob(bx, by, bz, rx * 0.86, rzz * 0.86, prnd, 12, 7); S.add(vv, tt, col=tuple(c * 0.45 for c in gk), mat=PINE)
        foliage(S, bx, by, bz + rzz * 0.15, rx, rzz, 1300, 0.21, gk, prnd, PINE)
        WEB_CLUMPS.append([round(bx, 2), round(by, 2), round(bz + rzz * 0.15, 2), round(rx, 2), round(rzz, 2)] + [round(c, 3) for c in gk])
        a0 = np.array([tx_, ty_, tz_ - 0.8]); b0 = np.array([bx, by, bz - rzz * 0.4]); sd = np.cross(b0 - a0, (0, 0, 1.0)); sd = sd / (np.linalg.norm(sd) + 1e-6) * 0.11
        S.add([a0 - sd, a0 + sd, b0 + sd * 0.5, b0 - sd * 0.5, a0 + (0, 0, 0.2), b0 + (0, 0, 0.12)], [(0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 4), (1, 4, 5), (1, 5, 2)], col=(1, 1, 1), mat=BARK)
S.save("site"); print("crossings", ncross, "junctions", len(JUNC)); print("planters", nplant, "small trees", ndec, "stelae", nstele); print("people", npeople, "signals", nsig, "far cars", nfar); print("site: panels", panels, "lamps", nl, "cars", ncar, "pines", len(pines), "lawn m2", round(lawn.area))

json.dump({"trees": WEB_TREES, "clumps": WEB_CLUMPS, "tower": [TX, TY, FLAT], "bounds": [x0, y0, x1, y1]}, open(f"{root}/data/web_trees.json", "w"))

# ================= Aristotelous Square =================
# Layout from the map (outline, 16 lawns as holes, the half-round plaza with its two crescents, the planters of the axis) and from
# aerial photographs (central grey field, slab borders, red patterned panels, black discs, white columns, palms, cafe tents, kiosk).
# Positions marked "est." are read off photographs by eye.
A = Mesh(); APAVE, ACENTRE, AGRASS, AMARB, AMATTE, ALEAF, ABARK, ALAMP, AGLOW, AASPH, ABAND, ACARPET, ARED, APALE, AMARK, ASIDE = range(16); AZ = FLAT; arnd = random.Random(17); RA = 300.0
abu = unary_union([p_ for p_ in polys if p_.distance(AC) < RA + 45]); aland = AC.buffer(RA, 64).difference(sea)
adrive = unary_union([LineString(r["line"]).buffer(r["w"] / 2, cap_style=2) for r in L["roads"] if r["kind"] in DRIVE and LineString(r["line"]).distance(AC) < RA + 20]).intersection(aland).difference(abu.buffer(0.2))
def polys_of(g): return [q for q in (g.geoms if hasattr(g, "geoms") else [g]) if q.geom_type == "Polygon" and not q.is_empty]
def aflat(poly, z, mat, col=(1, 1, 1)):
    for pp in polys_of(poly):
        if pp.area > 0.15: v, t = tri(pp); A.add(np.c_[v, np.full(len(v), AZ + z)], t, col=col, mat=mat)
def rect(a0, a1, t0, t1): return Polygon([Wd(a0, t0), Wd(a1, t0), Wd(a1, t1), Wd(a0, t1)])
def kerb(g, h=0.16, w=0.22, mat=AMARB):
    k = g.boundary.buffer(w); aflat(k, h, mat)
    for pp in polys_of(k):
        for ring in [pp.exterior] + list(pp.interiors):
            c = list(ring.coords)
            for p_, r_ in zip(c[:-1], c[1:]): A.add([(p_[0], p_[1], AZ), (r_[0], r_[1], AZ), (r_[0], r_[1], AZ + h), (p_[0], p_[1], AZ + h)], Q, mat=mat)
SEMI = next(unary_union([Polygon(p_[0], p_[1:]).buffer(0) for p_ in it["poly"]]) for it in L["plaza"] if not it.get("name") and Polygon(it["poly"][0][0]).distance(Point(Wd(70, 0))) < 5)
AXIS = _pl["Πεζόδρομος Αριστοτέλους"].intersection(AC.buffer(RA))
def holes_of(g): return [Polygon(r_.coords).buffer(0) for q in polys_of(g) for r_ in q.interiors]
H_SQ, H_SEMI, H_AX = holes_of(ASQ), holes_of(SEMI), [h for h in holes_of(_pl["Πεζόδρομος Αριστοτέλους"]) if h.distance(AC) < RA - 10]
lawns = unary_union(H_SQ + H_SEMI + H_AX)
sq_outer = Polygon(polys_of(ASQ)[0].exterior.coords); semi_outer = Polygon(polys_of(SEMI)[0].exterior.coords)
ped = unary_union([sq_outer, semi_outer, AXIS]).difference(abu.buffer(0.1)).difference(adrive).difference(lawns); pped = prep(ped)
# ---- ground: asphalt on every street, slab pavements everywhere else ----
aflat(aland.difference(abu.buffer(0.1)).difference(adrive).difference(sq_outer).difference(semi_outer).difference(AXIS), 0.06, ASIDE)
aflat(adrive, 0.03, AASPH); aflat(sea.buffer(1.2).intersection(aland), 0.09, AMARB)
centre = rect(-44, 40, -20, 20).difference(lawns.buffer(1.2))                                   # the plain grey field in the middle
aflat(centre, 0.065, ACENTRE); aflat(sq_outer.difference(lawns).difference(centre).difference(adrive), 0.06, APAVE)
aflat(centre.boundary.buffer(0.35), 0.07, AMARB)
aflat(semi_outer.difference(lawns).difference(adrive), 0.06, APALE); aflat(AXIS.difference(lawns).difference(abu.buffer(0.1)).difference(adrive), 0.06, APAVE)
def carpet(al, tl, size):                                                                      # red patterned panels (est.)
    hs = size / 2; c = [Wd(al - hs, tl - hs), Wd(al + hs, tl - hs), Wd(al + hs, tl + hs), Wd(al - hs, tl + hs)]
    if all(pped.contains(Point(q)) for q in c): A.add([(q[0], q[1], AZ + 0.075) for q in c], Q, uv=[(0, 0), (1, 0), (1, 1), (0, 1)], mat=ACARPET)
carpet(45.6, 0, 8.6)
for sd in (-1, 1): carpet(45.6, sd * 31.5, 6.0); carpet(-46.0, sd * 31.5, 5.0)
for sd in (-1, 1):
    for al in (-27.5, -2.0, 23.5): aflat(rect(al, al + 9, sd * 36.2, sd * 39.4).intersection(ped), 0.075, ARED)
for al in np.arange(92.0, 290.0, 6.0): aflat(rect(al, al + 0.55, -15, 15).intersection(ped), 0.075, ABAND)   # dark bands across the pedestrian street
# ---- trees, palms, lawns ----
def atree(x, y, z, H, col):
    limb(A, (x, y, z), (x + arnd.uniform(-.2, .2), y + arnd.uniform(-.2, .2), z + H * 0.55), 0.17, 0.11, (0.22, 0.16, 0.11), AMATTE, 7, False)
    for k in range(6):
        a = arnd.uniform(0, 6.28); dd = 0 if k == 0 else arnd.uniform(0.7, 1.7); bx, by, bz = x + dd * math.cos(a), y + dd * math.sin(a), z + H * arnd.uniform(0.6, 0.84); rx = arnd.uniform(1.5, 2.3)
        v, t = blob(bx, by, bz, rx * 0.86, rx * 0.68, arnd, 10, 6); A.add(v, t, col=tuple(k2 * 0.5 for k2 in col), mat=ALEAF); foliage(A, bx, by, bz, rx, rx * 0.8, 750, 0.2, col, arnd, ALEAF)
        limb(A, (x, y, z + H * 0.5), (bx, by, bz - rx * 0.3), 0.06, 0.03, (0.22, 0.16, 0.11), AMATTE, 5, False)
def green(): return (arnd.uniform(0.09, 0.15), arnd.uniform(0.22, 0.32), arnd.uniform(0.05, 0.08))
ntree = npalm = 0
def plant(g, kind):
    global ntree, npalm
    if g.is_empty or g.area < 4: return
    aflat(g.buffer(-0.2), 0.12, AGRASS); kerb(g); c = g.centroid; loc = (np.array([c.x, c.y]) - M0); al, tl = float(loc @ av), float(loc @ tv)
    if g.area < 22: atree(c.x, c.y, AZ + 0.12, arnd.uniform(5.5, 7.5), green()); ntree += 1; return
    if g.area < 45: palm(A, c.x, c.y, AZ + 0.12, arnd.uniform(3.5, 5.5), arnd, AMATTE, ALEAF); npalm += 1; return
    inner = g.buffer(-2.2); got = []
    if inner.is_empty: inner = g.buffer(-0.8)
    if kind == "semi":                                                                          # three palms along each crescent
        for q in [inner.exterior.interpolate(f, normalized=True) for f in (0.12, 0.45, 0.78)]: palm(A, q.x, q.y, AZ + 0.12, arnd.uniform(4.5, 6.5), arnd, AMATTE, ALEAF); npalm += 1
        return
    if kind == "axis":                                                                          # palms and clipped shrubs down the middle of the street
        mr = list(g.minimum_rotated_rectangle.exterior.coords); d0 = max((math.dist(mr[i], mr[i + 1]), mr[i], mr[i + 1]) for i in range(4)); n_ = max(1, int(d0[0] / 7))
        for i in range(n_):
            f = (i + 0.5) / n_; q = Point(c.x + (d0[2][0] - d0[1][0]) * (f - 0.5), c.y + (d0[2][1] - d0[1][1]) * (f - 0.5))
            if g.contains(q):
                if i % 2 == 0: palm(A, q.x, q.y, AZ + 0.12, arnd.uniform(4, 6.5), arnd, AMATTE, ALEAF); npalm += 1
                else:
                    v, t = blob(q.x, q.y, AZ + 1.2, 1.3, 1.1, arnd, 10, 6); A.add(v, t, col=(0.05, 0.13, 0.04), mat=ALEAF); foliage(A, q.x, q.y, AZ + 1.2, 1.45, 1.2, 500, 0.16, (0.08, 0.2, 0.05), arnd, ALEAF)
        return
    far = max(g.exterior.coords, key=lambda q: abs(float((np.array(q) - M0) @ av)))            # a palm at the end of each big lawn (est.), trees over the rest
    q = Point(far[0] + (c.x - far[0]) * 0.22, far[1] + (c.y - far[1]) * 0.22); palm(A, q.x, q.y, AZ + 0.12, arnd.uniform(5, 7), arnd, AMATTE, ALEAF); npalm += 1; got.append(q)
    bx = inner.bounds; want = max(1, int(g.area / 70)); tries = 0
    while len(got) < want + 1 and tries < 500:
        tries += 1; q = Point(arnd.uniform(bx[0], bx[2]), arnd.uniform(bx[1], bx[3]))
        if inner.contains(q) and all(q.distance(o) > 6 for o in got): got.append(q); atree(q.x, q.y, AZ + 0.12, arnd.uniform(6.5, 9.5), green()); ntree += 1
for g in H_SQ: plant(g, "sq")
for g in H_SEMI: plant(g, "semi")
for g in H_AX: plant(g, "axis")
# ---- lamps, banners, benches ----
def lamp_classic(q, u, arms=2):
    obox(A, q, u, 0.5, 0.5, AZ, AZ + 0.9, (1, 1, 1), ALAMP); limb(A, (q[0], q[1], AZ + 0.9), (q[0], q[1], AZ + 5.2), 0.11, 0.07, (0.03, 0.07, 0.05), AMATTE, 8, False)
    pts = [(0, 0, 5.75)] + [(0.75 * math.cos(k * 6.283 / arms), 0.75 * math.sin(k * 6.283 / arms), 5.0) for k in range(arms)]
    for dx, dy, dz in pts:
        c = (q[0] + u[0] * dx - u[1] * dy, q[1] + u[1] * dx + u[0] * dy, AZ + dz); limb(A, (q[0], q[1], AZ + 4.9), (c[0], c[1], c[2] - 0.25), 0.035, 0.03, (0.03, 0.07, 0.05), AMATTE, 5, False)
        v, t = blob(c[0], c[1], c[2], 0.2, 0.24, arnd, 8, 5); A.add(v, t, mat=AGLOW)
def lamp_tall(q, u):                                                                           # tall grey post with a banner
    limb(A, (q[0], q[1], AZ), (q[0], q[1], AZ + 10.5), 0.13, 0.07, (0.45, 0.46, 0.47), AMATTE, 8, False); obox(A, q, u, 0.9, 0.35, AZ + 10.5, AZ + 10.7, (0.45, 0.46, 0.47), AMATTE)
    obox(A, (q[0] + u[0] * 0.5, q[1] + u[1] * 0.5), u, 0.75, 0.03, AZ + 3.2, AZ + 5.3, (0.15, 0.45, 0.70), AMATTE); obox(A, (q[0] + u[0] * 0.5, q[1] + u[1] * 0.5), u, 0.55, 0.04, AZ + 3.6, AZ + 4.6, (0.85, 0.88, 0.9), AMATTE)
def bench(q, u, col=(0.62, 0.08, 0.06)):
    n = (-u[1], u[0])
    for k in range(4): obox(A, (q[0] + n[0] * (k - 1.5) * 0.11, q[1] + n[1] * (k - 1.5) * 0.11), u, 1.8, 0.09, AZ + 0.44, AZ + 0.48, col, AMATTE)
    for k in range(3): obox(A, (q[0] - n[0] * 0.26, q[1] - n[1] * 0.26), u, 1.8, 0.035, AZ + 0.56 + k * 0.13, AZ + 0.65 + k * 0.13, col, AMATTE)
    for e in (-0.75, 0.75): obox(A, (q[0] + u[0] * e, q[1] + u[1] * e), n, 0.5, 0.05, AZ, AZ + 0.44, (0.05, 0.05, 0.05), AMATTE); obox(A, (q[0] + u[0] * e - n[0] * 0.26, q[1] + u[1] * e - n[1] * 0.26), n, 0.05, 0.05, AZ + 0.44, AZ + 0.95, (0.05, 0.05, 0.05), AMATTE)
ua, ut = (float(av[0]), float(av[1])), (float(tv[0]), float(tv[1])); nlamp = 0
for sd in (-1, 1):
    for al in np.arange(-38, 39, 19.0):
        q = Wd(al, sd * 21.2)
        if pped.contains(Point(q)): lamp_tall(q, ut if sd < 0 else (-ut[0], -ut[1])); nlamp += 1
    for al, tl in ((47.8, 8), (47.8, 26), (47.8, 40), (-48.5, 8), (-48.5, 26), (-48.5, 40), (0, 44.5), (25, 44.5), (-25, 44.5)):
        q = Wd(al, sd * tl)
        if pped.contains(Point(q)): lamp_classic(q, ua, 2); nlamp += 1
    for al in np.arange(96, 290, 19.0):
        q = Wd(al, sd * 10.5)
        if pped.contains(Point(q)) and abu.distance(Point(q)) > 3: lamp_classic(q, ua, 2); nlamp += 1
nbench = 0
for tl in np.arange(-16.5, 17, 5.5):
    q = Wd(38.2, tl); bench(q, ut); nbench += 1; continue                                                           # red benches along the north edge of the grey field (est.)
    q = Wd(42.2, tl)
    if pped.contains(Point(q)): bench(q, ut); nbench += 1
for sd in (-1, 1):
    for al in np.arange(-36, 37, 12.0):
        q = Wd(al, sd * 18.6)
        if True: bench(q, ua if sd > 0 else (-ua[0], -ua[1])); nbench += 1
# ---- the half-round plaza: black discs with a lamp, the row of white advertising columns ----
for g in H_SEMI:
    c = g.centroid; loc = np.array([c.x, c.y]) - M0; sgn = 1 if float(loc @ tv) > 0 else -1; q = Wd(65.6, sgn * 16.5)               # est.
    aflat(Point(q).buffer(3.9, 24).intersection(ped), 0.08, ABAND); aflat(Point(q).buffer(4.15, 24).difference(Point(q).buffer(3.9, 24)).intersection(ped), 0.085, AMARB); lamp_classic(q, ua, 4)
for k in range(6):
    q = Wd(67.0, -10.5 + k * 4.2); limb(A, (q[0], q[1], AZ), (q[0], q[1], AZ + 2.5), 0.62, 0.62, (0.92, 0.92, 0.90), AMATTE, 16); limb(A, (q[0], q[1], AZ + 2.5), (q[0], q[1], AZ + 2.62), 0.68, 0.68, (0.2, 0.2, 0.2), AMATTE, 16)
    limb(A, (q[0], q[1], AZ + 0.7), (q[0], q[1], AZ + 2.0), 0.63, 0.63, [(0.85, 0.2, 0.3), (0.2, 0.3, 0.6), (0.95, 0.75, 0.1), (0.1, 0.1, 0.1), (0.3, 0.6, 0.5), (0.6, 0.2, 0.6)][k], AMATTE, 16, False)
# ---- road markings, crossings, bus shelters, bollards, buses ----
def paintA(p0, p1, wd): 
    d = np.array(p1) - np.array(p0); n_ = np.linalg.norm(d)
    if n_ < 0.1: return
    s_ = np.array([-d[1], d[0]]) / n_ * wd / 2; A.add([(p0[0] - s_[0], p0[1] - s_[1], AZ + 0.045), (p1[0] - s_[0], p1[1] - s_[1], AZ + 0.045), (p1[0] + s_[0], p1[1] + s_[1], AZ + 0.045), (p0[0] + s_[0], p0[1] + s_[1], AZ + 0.045)], Q, mat=AMARK)
for al0, lanes, wd_ in ((57.9, 3, 11.0), (-56.0, 4, 14.0)):                                    # Mitropoleos and Nikis: lane lines along the square
    for k in range(lanes + 1):
        al = al0 - wd_ / 2 + 0.2 + k * (wd_ - 0.4) / lanes
        for tl in np.arange(-150, 150, 9.0 if 0 < k < lanes else 2.0):
            p0, p1 = Wd(al, tl), Wd(al, tl + (3.0 if 0 < k < lanes else 2.0))
            if adrive.contains(Point(p0)) and adrive.contains(Point(p1)) and not any(abs(tl - zc) < 4.5 for zc in (-30, 0, 30)): paintA(p0, p1, 0.15)
    for zc in (-30, 0, 30):                                                                    # zebra crossings
        for k in range(int(wd_ - 0.8)):
            al = al0 - wd_ / 2 + 0.6 + k; p0, p1 = Wd(al, zc - 2.0), Wd(al, zc + 2.0)
            if adrive.contains(Point(p0)) and adrive.contains(Point(p1)): paintA(p0, p1, 0.5)
for al, sgn in ((51.6, -1), (64.2, 1), (-48.3, 1)):                                            # bollards along the kerbs
    for tl in np.arange(-44, 45, 2.2):
        q = Wd(al, tl)
        if pped.contains(Point(q)) and not any(abs(tl - zc) < 3 for zc in (-30, 0, 30)): limb(A, (q[0], q[1], AZ), (q[0], q[1], AZ + 0.85), 0.06, 0.06, (0.75, 0.75, 0.72), AMATTE, 6)
for tl in (9.5, 17.0):                                                                         # bus shelters on Mitropoleos (est.)
    q = Wd(50.3, tl); fr = (0.10, 0.30, 0.22)
    for e in (-1.9, 1.9):
        for o in (-0.6, 0.6): obox(A, (q[0] + ut[0] * e + ua[0] * o, q[1] + ut[1] * e + ua[1] * o), ut, 0.08, 0.08, AZ, AZ + 2.4, fr, AMATTE)
    obox(A, q, ut, 4.2, 1.6, AZ + 2.4, AZ + 2.52, fr, AMATTE); obox(A, (q[0] - ua[0] * 0.6, q[1] - ua[1] * 0.6), ut, 3.8, 0.03, AZ + 0.4, AZ + 2.3, (0.35, 0.45, 0.48), AMATTE); obox(A, (q[0] - ua[0] * 0.3, q[1] - ua[1] * 0.3), ut, 2.4, 0.4, AZ + 0.42, AZ + 0.48, fr, AMATTE)
for al, tl, sg in ((-58.5, -22, 1), (-54.5, 38, 1), (59.5, -60, -1)):                           # city buses
    q = Wd(al, tl); u = (ut[0] * sg, ut[1] * sg)
    if adrive.contains(Point(q)):
        obox(A, q, u, 12, 2.55, AZ + 0.35, AZ + 3.1, (0.10, 0.25, 0.55), AMATTE); obox(A, q, u, 12.02, 2.57, AZ + 1.25, AZ + 2.35, (0.03, 0.04, 0.05), AMATTE); obox(A, q, u, 11.6, 2.4, AZ + 3.1, AZ + 3.3, (0.85, 0.85, 0.85), AMATTE)
        for e in (-4, 3.6):
            for o in (-1.2, 1.2): limb(A, (q[0] + u[0] * e - u[1] * (o - 0.12 * np.sign(o)), q[1] + u[1] * e + u[0] * (o - 0.12 * np.sign(o)), AZ + 0.5), (q[0] + u[0] * e - u[1] * o, q[1] + u[1] * e + u[0] * o, AZ + 0.5), 0.5, 0.5, (0.03, 0.03, 0.03), AMATTE, 10)
# ---- kiosk in the middle of the square (est.) ----
q = Wd(-33.5, 0); obox(A, q, ua, 3.4, 3.4, AZ, AZ + 2.5, (0.08, 0.16, 0.12), AMATTE); obox(A, q, ua, 3.42, 3.42, AZ + 1.0, AZ + 2.0, (0.05, 0.06, 0.07), AMATTE); obox(A, q, ua, 4.8, 4.8, AZ + 2.5, AZ + 2.62, (0.08, 0.16, 0.12), AMATTE)
cs = [(q[0] + (sa * ua[0] + sb * ut[0]) * 2.0, q[1] + (sa * ua[1] + sb * ut[1]) * 2.0, AZ + 2.62) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]; A.add(cs + [(q[0], q[1], AZ + 3.5)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], col=(0.10, 0.20, 0.15), mat=AMATTE)
# ---- cafes: white tent canopies in rows in front of the arcades, tables and chairs under them ----
TENT = [(0.92, 0.91, 0.87), (0.90, 0.88, 0.80), (0.88, 0.86, 0.84), (0.55, 0.42, 0.30), (0.92, 0.91, 0.87)]; ntent = nchair = 0
def tent(q, u, col):
    global nchair
    x, y = q; z = AZ + 0.06; hs = 2.0; n = (-u[1], u[0])
    for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)): limb(A, (x + (sa * u[0] + sb * n[0]) * hs * 0.96, y + (sa * u[1] + sb * n[1]) * hs * 0.96, z), (x + (sa * u[0] + sb * n[0]) * hs * 0.96, y + (sa * u[1] + sb * n[1]) * hs * 0.96, z + 2.5), 0.035, 0.035, (0.9, 0.9, 0.9), AMATTE, 5, False)
    cs = [(x + (sa * u[0] + sb * n[0]) * hs, y + (sa * u[1] + sb * n[1]) * hs, z + 2.5) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]; lo = [(c[0], c[1], z + 2.25) for c in cs]
    A.add(cs + [(x, y, z + 3.35)] + lo, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (0, 1, 6), (0, 6, 5), (1, 2, 7), (1, 7, 6), (2, 3, 8), (2, 8, 7), (3, 0, 5), (3, 5, 8), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)], col=col, mat=AMATTE)
    for sa, sb in ((-0.95, -0.95), (0.95, 0.95), (-0.95, 0.95), (0.95, -0.95)):
        tq = (x + sa * u[0] + sb * n[0], y + sa * u[1] + sb * n[1]); obox(A, tq, u, 0.7, 0.7, z + 0.71, z + 0.74, (0.32, 0.24, 0.16), AMATTE); limb(A, (tq[0], tq[1], z), (tq[0], tq[1], z + 0.71), 0.04, 0.03, (0.1, 0.1, 0.1), AMATTE, 5, False)
        cc = (0.50, 0.40, 0.26) if arnd.random() < 0.7 else (0.15, 0.15, 0.16)
        for ca, cb, fu in ((0.62, 0, (-u[0], -u[1])), (-0.62, 0, u), (0, 0.62, (-n[0], -n[1])), (0, -0.62, n)):
            if arnd.random() < 0.8: chair(A, (tq[0] + ca * u[0] + cb * n[0], tq[1] + ca * u[1] + cb * n[1]), fu, z, cc, AMATTE); nchair += 1
for sd in (-1, 1):
    for row, tl in enumerate((37.6, 41.9)):
        for i, al in enumerate(np.arange(-42, 43, 4.3)):
            if i % 6 == 5: continue
            q = Wd(al, sd * tl)
            if pped.contains(Point(q)) and lawns.distance(Point(q)) > 2.3 and abu.distance(Point(q)) > 1.5: tent(q, ua, TENT[(i // 6 + (sd > 0)) % len(TENT)]); ntent += 1
bnd = ped.buffer(-5.5)                                                                         # and along the arcades of the pedestrian street and the half-round plaza
for ring in [g_.exterior for g_ in polys_of(bnd)]:
    for i in range(int(ring.length / 4.6)):
        q = ring.interpolate(i * 4.6); q2 = ring.interpolate(i * 4.6 + 0.5); u = (q2.x - q.x, q2.y - q.y); n_ = math.hypot(*u) or 1; u = (u[0] / n_, u[1] / n_); loc = np.array([q.x, q.y]) - M0
        if float(loc @ av) < 62 or q.distance(AC) > RA - 15 or abu.distance(q) > 8 or abu.distance(q) < 3.5 or adrive.distance(q) < 4 or lawns.distance(q) < 2.5 or i % 7 == 6: continue
        tent((q.x, q.y), u, TENT[(i // 7) % len(TENT)]); ntent += 1
# ---- Aristotle: bronze seated figure on a marble plinth (the position is an estimate) ----
sq = Wd(-5.0, 23.5); BR = (0.14, 0.11, 0.07); f_ = np.array([-tv[0], -tv[1], 0.0]); s_ = np.array([av[0], av[1], 0.0]); zu = np.array([0, 0, 1.0]); O = np.array([sq[0], sq[1], AZ])
obox(A, sq, ut, 2.4, 2.4, AZ, AZ + 0.3, (1, 1, 1), AMARB); obox(A, sq, ut, 1.9, 1.9, AZ + 0.3, AZ + 1.6, (1, 1, 1), AMARB); obox(A, sq, ut, 2.1, 2.1, AZ + 1.6, AZ + 1.75, (1, 1, 1), AMARB)
def SP(fw, sd, h): return O + f_ * fw + s_ * sd + zu * (1.75 + h)
obox(A, (sq[0] - f_[0] * 0.25, sq[1] - f_[1] * 0.25), (float(f_[0]), float(f_[1])), 0.9, 1.0, AZ + 1.75, AZ + 2.35, BR, AMATTE)
for sg in (-1, 1):
    limb(A, SP(-0.1, sg * 0.2, 0.72), SP(0.5, sg * 0.24, 0.70), 0.15, 0.12, BR, AMATTE); limb(A, SP(0.5, sg * 0.24, 0.70), SP(0.62, sg * 0.24, 0.08), 0.11, 0.08, BR, AMATTE); limb(A, SP(0.62, sg * 0.24, 0.05), SP(0.9, sg * 0.24, 0.03), 0.07, 0.06, BR, AMATTE)
    limb(A, SP(-0.15, sg * 0.34, 1.52), SP(0.05, sg * 0.4, 1.12), 0.09, 0.075, BR, AMATTE); limb(A, SP(0.05, sg * 0.4, 1.12), SP(0.42, sg * 0.3, 0.92), 0.07, 0.055, BR, AMATTE)
limb(A, SP(-0.18, 0, 0.6), SP(-0.12, 0, 1.6), 0.26, 0.28, BR, AMATTE, 10); limb(A, SP(-0.12, 0, 1.6), SP(-0.08, 0, 1.75), 0.09, 0.08, BR, AMATTE)
c = SP(-0.05, 0, 1.92); v, t = blob(c[0], c[1], c[2], 0.17, 0.2, arnd, 10, 6); A.add(v, t, col=BR, mat=AMATTE)
# ---- people and pigeons ----
wb = ped.bounds; nap = 0; tries = 0
while nap < 460 and tries < 60000:
    tries += 1; x, y = arnd.uniform(wb[0], wb[2]), arnd.uniform(wb[1], wb[3])
    if not pped.contains(Point(x, y)) or math.hypot(x - ACX, y - ACY) > RA - 20 or abu.distance(Point(x, y)) < 0.8: continue
    a = arnd.uniform(0, 6.28) if arnd.random() < 0.5 else math.atan2(av[1], av[0]) + (0 if arnd.random() < 0.5 else math.pi) + arnd.uniform(-0.4, 0.4)
    person2(A, x, y, AZ + 0.07, a, 0.62 if arnd.random() < 0.08 else arnd.uniform(0.93, 1.07), arnd, AMATTE); nap += 1
    if arnd.random() < 0.45 and pped.contains(Point(x - math.sin(a) * 0.62, y + math.cos(a) * 0.62)): person2(A, x - math.sin(a) * 0.62, y + math.cos(a) * 0.62, AZ + 0.07, a, arnd.uniform(0.9, 1.05), arnd, AMATTE); nap += 1
for _ in range(70):
    q = Wd(arnd.uniform(-35, 35), arnd.uniform(-17, 17)); v, t = blob(q[0], q[1], AZ + 0.16, 0.13, 0.08, arnd, 6, 4); A.add(v, t, col=(0.32, 0.33, 0.36), mat=AMATTE)
    v, t = blob(q[0] + 0.1, q[1], AZ + 0.24, 0.045, 0.045, arnd, 5, 3); A.add(v, t, col=(0.25, 0.27, 0.32), mat=AMATTE)
A.save("aris"); json.dump({"x": ACX, "y": ACY, "z": AZ, "a": [float(av[0]), float(av[1])], "La": La, "Wt": Wt, "m": [float(M0[0]), float(M0[1])], "R": RA}, open(f"{root}/data/aris_site.json", "w"))
print("aris: trees", ntree, "palms", npalm, "lamps", nlamp, "benches", nbench, "tents", ntent, "chairs", nchair, "people", nap)
