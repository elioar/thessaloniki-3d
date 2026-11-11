"""Stage 7: the White Tower, built in code.
Published sizes: drum 23 m across at the base and 27 m high, turret 12 m across, about 34 m in all.
Read off the reference photos (estimates): the batter of the drum, the string course at ~0.6 of the drum height,
56 machicolation arches, 44 brick merlons on the drum and 22 on the turret, the window rows, the mast."""
import json, os, math, random
import numpy as np, mapbox_earcut as earcut
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = dict(base_r=11.5, cordon_z=15.6, cordon_r=10.75, top_r=10.45, mach_z=22.4, walk_z=24.05, drum_top=27.0, arches=56, merlons=44,
         turret_r=6.0, turret_walk=31.0, turret_top=33.6, turret_merlons=22, door_deg=85.0, strip=0.4)
STONE, DARK, PAVE, WHITE, BLUE, METAL, BRICK, CAP = range(8)
V = []; T = []; UV = []; M = []; n = 0
def add(verts, tris, uv, mat):
    global n
    V.append(np.asarray(verts, float)); T.append(np.asarray(tris, np.int64) + n); UV.append(np.asarray(uv, float)); M.append(np.full(len(tris), mat)); n += len(verts)
def drum_r(z):
    if z <= P["cordon_z"]: return P["base_r"] + (P["cordon_r"] - P["base_r"]) * z / P["cordon_z"]
    return P["cordon_r"] + (P["top_r"] - P["cordon_r"]) * min(1, (z - P["cordon_z"]) / (P["mach_z"] - P["cordon_z"]))
def arch(cu, z0, w, h, k=10, flat=False):
    if flat: return box(cu - w / 2, z0, cu + w / 2, z0 + h)
    pts = [(cu - w / 2, z0), (cu + w / 2, z0)]
    for i in range(k + 1):
        a = math.pi * i / k; pts.append((cu + w / 2 * math.cos(a), z0 + h - w / 2 + w / 2 * math.sin(a)))
    return Polygon(pts)
def tri2d(poly):
    rings = [np.array(poly.exterior.coords[:-1])] + [np.array(r.coords[:-1]) for r in poly.interiors]
    v = np.vstack(rings); return v, earcut.triangulate_float64(v, np.cumsum([len(r) for r in rings]).astype(np.uint32)).reshape(-1, 3)
def wall(rf, r_ref, z0, z1, openings, mat=STONE, reveal=0.8, back=DARK, vstep=1.0):
    """wall of revolution with real openings: cut flat in (u, v), in narrow strips, then bent. rf(z) gives the radius."""
    C = 2 * math.pi * r_ref; holes = unary_union(openings) if openings else None
    def pt(u, v, d=0.0): r = rf(v) - d; th = u / r_ref; return (r * math.cos(th), r * math.sin(th), v)
    ns = int(round(C / P["strip"])); du = C / ns; nv = max(1, int(round((z1 - z0) / vstep)))
    for i in range(ns):
        for j in range(nv):
            cell = box(i * du, z0 + (z1 - z0) * j / nv, (i + 1) * du, z0 + (z1 - z0) * (j + 1) / nv); part = cell.difference(holes) if holes is not None else cell
            for pp in (part.geoms if hasattr(part, "geoms") else [part]):
                if pp.geom_type != "Polygon" or pp.area < 1e-6: continue
                v, t = tri2d(pp); add([pt(a, b) for a, b in v], t[:, ::-1], v, mat)
    for o in openings:
        c = list(o.exterior.coords)
        for (a, b), (e, f) in zip(c[:-1], c[1:]):
            add([pt(a, b), pt(e, f), pt(e, f, reveal), pt(a, b, reveal)], [(0, 1, 2), (0, 2, 3)], [(a, b), (e, f), (e + reveal, f), (a + reveal, b)], mat)
        v, t = tri2d(o); add([pt(a, b, reveal) for a, b in v], t[:, ::-1], v, back)
def ring(ri, ro, z0, z1, a0=0.0, a1=2 * math.pi, mat=STONE, ro1=None, ri1=None):
    """curved block between two angles; ro1/ri1 let the top be narrower or wider (slopes, mouldings)."""
    ro1 = ro if ro1 is None else ro1; ri1 = ri if ri1 is None else ri1
    k = max(1, int(round((a1 - a0) * ro / 0.4))); full = abs(a1 - a0 - 2 * math.pi) < 1e-9
    A = [a0 + (a1 - a0) * i / k for i in range(k + 1)]
    def p(r, a, z): return (r * math.cos(a), r * math.sin(a), z)
    def q(p0, p1, p2, p3, uv): add([p0, p1, p2, p3], [(0, 1, 2), (0, 2, 3)], uv, mat)
    for a, b in zip(A[:-1], A[1:]):
        q(p(ro, a, z0), p(ro, b, z0), p(ro1, b, z1), p(ro1, a, z1), [(a * ro, z0), (b * ro, z0), (b * ro, z1), (a * ro, z1)])
        if ri > 0: q(p(ri, b, z0), p(ri, a, z0), p(ri1, a, z1), p(ri1, b, z1), [(b * ri, z0), (a * ri, z0), (a * ri, z1), (b * ri, z1)])
        q(p(ro1, a, z1), p(ro1, b, z1), p(ri1, b, z1), p(ri1, a, z1), [(a * ro, ro1), (b * ro, ro1), (b * ro, ri1), (a * ro, ri1)])
        q(p(ro, b, z0), p(ro, a, z0), p(ri, a, z0), p(ri, b, z0), [(b * ro, ro), (a * ro, ro), (a * ro, ri), (b * ro, ri)])
    if not full:
        for a, flip in ((a0, False), (a1, True)):
            pts = [p(ri, a, z0), p(ro, a, z0), p(ro1, a, z1), p(ri1, a, z1)]
            add(pts[::-1] if flip else pts, [(0, 1, 2), (0, 2, 3)], [(ri, z0), (ro, z0), (ro, z1), (ri, z1)], mat)
def battlement(r_out, z_base, z_sill, z_top, count, thick, duty=0.60):
    """brick parapet with merlons and stone caps that slope inward."""
    ring(r_out - thick, r_out, z_base, z_sill, mat=BRICK); d = 2 * math.pi / count
    for i in range(count):
        a0, a1 = i * d + d * (1 - duty) / 2, i * d + d * (1 + duty) / 2
        ring(r_out - thick, r_out, z_sill, z_top, a0, a1, mat=BRICK)
        ring(r_out - thick - 0.04, r_out + 0.05, z_top, z_top + 0.22, a0 - 0.004, a1 + 0.004, mat=CAP, ro1=r_out - thick * 0.35, ri1=r_out - thick * 0.75)
    for i in range(count):                             # low sill stones in the gaps
        a0, a1 = i * d + d * (1 + duty) / 2, (i + 1) * d + d * (1 - duty) / 2
        ring(r_out - thick - 0.03, r_out + 0.03, z_sill, z_sill + 0.08, a0, a1, mat=CAP)
rnd = random.Random(3); R = P["base_r"]; C = 2 * math.pi * R; door_u = math.radians(P["door_deg"]) * R
# ---- drum openings: few, small, irregular (as in the photos) ----
op = [arch(door_u, 0.0, 1.9, 3.3)]
def row(z, count, w, h, phase, jit=0.6, flat_every=0, skip=()):
    for i in range(count):
        if i in skip: continue
        u = (door_u + C * (i + phase) / count + rnd.uniform(-jit, jit)) % C
        if w < u < C - w: op.append(arch(u, z + rnd.uniform(-0.25, 0.25), w, h, flat=bool(flat_every and i % flat_every == 0)))
row(19.6, 10, 0.75, 1.25, 0.5, flat_every=3)          # upper row, under the machicolations
row(16.9, 6, 0.7, 1.2, 0.15, flat_every=2)            # just above the string course
row(12.9, 5, 0.85, 1.45, 0.6)                         # below the string course
row(11.6, 7, 0.14, 1.3, 0.1, flat_every=1)            # arrow slits
row(7.4, 4, 0.9, 1.6, 0.35, flat_every=1)             # big square windows low down
row(3.4, 5, 0.6, 0.9, 0.5, flat_every=1, skip=(4,))
wall(drum_r, R, 0.0, P["mach_z"], op)
zc = P["cordon_z"]; rc = P["cordon_r"]                # string course: a rounded moulding
ring(rc - 0.05, rc + 0.12, zc - 0.20, zc - 0.08, ro1=rc + 0.24); ring(rc - 0.05, rc + 0.24, zc - 0.08, zc + 0.08); ring(rc - 0.05, rc + 0.24, zc + 0.08, zc + 0.20, ro1=rc + 0.10)
# ---- machicolations: stone corbels carrying small arches, standing 0.55 m proud of the wall ----
rt = P["top_r"]; ro = rt + 0.55; na = P["arches"]; Cm = 2 * math.pi * ro; pa = Cm / na; z0 = P["mach_z"]; zs = z0 + 1.55
arches = [arch((i + 0.5) * pa, z0 - 0.01, pa * 0.66, 1.22) for i in range(na)]
wall(lambda z: ro, ro, z0, zs, arches, mat=STONE, reveal=0.55, back=STONE, vstep=2.0)
da = 2 * math.pi / na
for i in range(na):                                    # two-step corbel under every pier
    ring(rt - 0.05, rt + 0.22, z0 - 0.62, z0 - 0.30, i * da - da * 0.13, i * da + da * 0.13, mat=CAP)
    ring(rt - 0.05, rt + 0.42, z0 - 0.30, z0, i * da - da * 0.15, i * da + da * 0.15, mat=CAP)
    ring(rt, ro, z0, z0 + 0.02, i * da - da * 0.17, i * da + da * 0.17)          # underside of the pier
ring(rt - 0.05, ro + 0.06, zs, zs + 0.16, mat=CAP)                             # stone band over the arches
wall(lambda z: rt, rt, P["mach_z"], zs, [])                                    # wall behind the arches
battlement(ro, zs + 0.16, P["walk_z"] + 0.95, P["drum_top"] - 0.22, P["merlons"], 0.55)
ring(0.0, ro - 0.55, P["walk_z"] - 0.3, P["walk_z"], mat=PAVE)                 # the walk
# ---- turret ----
r2 = P["turret_r"]; C2 = 2 * math.pi * r2; zt = P["walk_z"]; zt1 = P["turret_walk"] - 0.55
op2 = [arch(math.radians(P["door_deg"] + 200) * r2 % C2, zt, 1.1, 2.2), arch(math.radians(P["door_deg"] + 20) * r2, zt + 0.9, 1.0, 1.0, flat=True)]
for i in range(6): op2.append(arch((C2 * (i + 0.5) / 6 + 1.9) % (C2 - 1.4) + 0.7, zt + 2.6 + (i % 2) * 0.5, 0.7, 1.35))
wall(lambda z: r2, r2, zt, zt1, op2, reveal=0.6)
ring(r2 - 0.05, r2 + 0.10, zt1, zt1 + 0.14, ro1=r2 + 0.26, mat=CAP); ring(r2 - 0.05, r2 + 0.26, zt1 + 0.14, zt1 + 0.34, mat=CAP)   # cornice band
battlement(r2 + 0.26, zt1 + 0.34, P["turret_walk"] + 0.75, P["turret_top"] - 0.22, P["turret_merlons"], 0.5)
ring(0.0, r2 - 0.24, P["turret_walk"] - 0.3, P["turret_walk"], mat=PAVE)
# ---- mast, guy wires, cross, flag ----
zr = P["turret_walk"]; mh = 10.5
ring(0, 0.45, zr, zr + 0.4, mat=CAP); ring(0, 0.09, zr + 0.4, zr + mh, mat=METAL, ro1=0.05)
ring(0, 0.03, zr + mh, zr + mh + 0.9, mat=METAL)
def bar(p, q, w, mat=METAL):
    p, q = np.array(p, float), np.array(q, float); d = q - p; s = np.cross(d, (0.3, 0.5, 0.81)); s = s / np.linalg.norm(s) * w; t = np.cross(d, s); t = t / np.linalg.norm(t) * w
    c = [p + s, p + t, p - s, p - t, q + s, q + t, q - s, q - t]
    add(c, [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)], [(0, 0)] * 8, mat)
bar((-0.3, 0, zr + mh + 0.6), (0.3, 0, zr + mh + 0.6), 0.03)
for k in range(6):
    a = k * math.pi / 3 + 0.3; bar((0, 0, zr + mh * 0.62), ((r2 - 0.5) * math.cos(a), (r2 - 0.5) * math.sin(a), zr + 0.9), 0.012)
fw, fh = 3.3, 2.2; ft = zr + mh - 0.3; sh = fh / 9; nx = 26
def fpt(s, t): return (0.09 + s * 0.94, 0.30 * math.sin(s * 2.4) * (s / fw) + 0.08 * math.sin(s * 5.5 + t * 1.5), ft - t - 0.5 * (s / fw) ** 2)
def fquad(s0, s1, t0, t1, mat):
    k = max(1, int((s1 - s0) / fw * nx))
    for i in range(k):
        a, b = s0 + (s1 - s0) * i / k, s0 + (s1 - s0) * (i + 1) / k
        add([fpt(a, t1), fpt(b, t1), fpt(b, t0), fpt(a, t0)], [(0, 1, 2), (0, 2, 3)], [(a, t1), (b, t1), (b, t0), (a, t0)], mat)
cs = 5 * sh
for i in range(9): fquad(cs if i < 5 else 0, fw, i * sh, (i + 1) * sh, BLUE if i % 2 == 0 else WHITE)
for (a, b, c, d, m) in [(0, 2 * sh, 0, 2 * sh, BLUE), (3 * sh, cs, 0, 2 * sh, BLUE), (0, 2 * sh, 3 * sh, cs, BLUE), (3 * sh, cs, 3 * sh, cs, BLUE),
                        (2 * sh, 3 * sh, 0, cs, WHITE), (0, 2 * sh, 2 * sh, 3 * sh, WHITE), (3 * sh, cs, 2 * sh, 3 * sh, WHITE)]: fquad(a, b, c, d, m)
v = np.vstack(V); t = np.vstack(T); m = np.concatenate(M)
np.savez(f"{root}/data/geo_tower.npz", v=v.astype(np.float32), t=t.astype(np.int32), uv=np.vstack(UV).astype(np.float32), col=np.ones((len(v), 3), np.float32), mat=m.astype(np.int32))
json.dump(P, open(f"{root}/data/tower_params.json", "w"))
# ---- checks ----
masonry = np.unique(t[np.isin(m, (STONE, BRICK, CAP))]); rad = np.hypot(v[masonry, 0], v[masonry, 1])
print("tower verts", len(v), "tris", len(t), "| masonry height", round(float(v[masonry, 2].max()), 2), "| base radius", round(float(rad[v[masonry, 2] < 0.01].max()), 2),
      "| crown radius", round(float(rad[v[masonry, 2] > 26].max()), 2), "| drum openings", len(op), "| arches", len(arches))
assert abs(v[masonry, 2].max() - P["turret_top"]) < 0.05 and len(arches) == 56
