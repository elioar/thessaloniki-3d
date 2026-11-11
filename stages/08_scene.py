"""Stage 8 (runs inside Blender, headless): load arrays, add materials, light, cameras; save .blend and render."""
import bpy, json, os, sys, math
import numpy as np
from mathutils import Vector
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(k, d): return argv[argv.index(k) + 1] if k in argv else d
CAMS = [c for c in arg("--cams", "aerial").split(",") if c and c != "none"]; RESX = int(arg("--resx", "1280")); SAMPLES = int(arg("--samples", "32")); CLAY = "--clay" in argv
ONLY_TOWER = "--only-tower" in argv
bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
# ---------------- helpers ----------------
def load(name, smooth=False):
    d = np.load(f"{root}/data/geo_{name}.npz"); v, t = d["v"], d["t"]
    me = bpy.data.meshes.new(name); me.vertices.add(len(v)); me.vertices.foreach_set("co", v.ravel())
    me.loops.add(t.size); me.loops.foreach_set("vertex_index", t.ravel())
    me.polygons.add(len(t)); me.polygons.foreach_set("loop_start", np.arange(0, t.size, 3, dtype=np.int32))
    me.polygons.foreach_set("material_index", d["mat"].astype(np.int32)); me.update()
    me.uv_layers.new(name="UVMap").uv.foreach_set("vector", d["uv"][t.ravel()].ravel())
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT"); ca.data.foreach_set("color", np.c_[d["col"], np.ones(len(v))].ravel().astype(np.float32))
    me.polygons.foreach_set("use_smooth", np.full(len(t), smooth)); me.update()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); return ob
def mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"); return m, nt, b
def N(nt, typ, **kw):
    n = nt.nodes.new(typ)
    for k, v in kw.items(): setattr(n, k, v)
    return n
def M(nt, op, a, b=None, c=None):
    n = nt.nodes.new("ShaderNodeMath"); n.operation = op
    for i, x in enumerate((a, b, c)):
        if x is None: continue
        if isinstance(x, (int, float)): n.inputs[i].default_value = x
        else: nt.links.new(x, n.inputs[i])
    return n.outputs[0]
def mix(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix"); n.data_type = "RGBA"
    for sock, x in ((n.inputs[0], fac), (n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(x, (int, float)): sock.default_value = x
        elif isinstance(x, tuple): sock.default_value = (*x, 1)
        else: nt.links.new(x, sock)
    return n.outputs[2]
def band(nt, x, lo, hi): return M(nt, "MULTIPLY", M(nt, "GREATER_THAN", x, lo), M(nt, "LESS_THAN", x, hi))
ASSETS = json.load(open(f"{root}/data/assets/assets.json"))
def pbr(nt, b, key, vec, scale, tint=None, tint_fac=0.0, bump=1.0):
    """wire a Poly Haven texture set (colour, roughness, normal) into a Principled BSDF; returns the colour socket."""
    mp = N(nt, "ShaderNodeMapping"); mp.inputs["Scale"].default_value = (scale, scale, scale); nt.links.new(vec, mp.inputs[0]); A = ASSETS[key]; col = None
    for kind in ("diff", "rough", "nor"):
        if kind not in A: continue
        t = N(nt, "ShaderNodeTexImage"); t.image = bpy.data.images.load(f"{root}/{A[kind]}", check_existing=True); nt.links.new(mp.outputs[0], t.inputs[0])
        if kind == "diff": col = t.outputs[0]
        else:
            t.image.colorspace_settings.name = "Non-Color"
            if kind == "rough": nt.links.new(t.outputs[0], b.inputs["Roughness"])
            else:
                nm = N(nt, "ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = bump; nt.links.new(t.outputs[0], nm.inputs["Color"]); nt.links.new(nm.outputs[0], b.inputs["Normal"])
    if tint is not None: col = mix(nt, tint_fac, col, tint) if tint_fac < 1 else col
    return col
def mul(nt, a, b_, fac=1.0):
    n = nt.nodes.new("ShaderNodeMix"); n.data_type = "RGBA"; n.blend_type = "MULTIPLY"; n.inputs[0].default_value = fac
    for sock, x in ((n.inputs[6], a), (n.inputs[7], b_)):
        if isinstance(x, tuple): sock.default_value = (*x, 1)
        else: nt.links.new(x, sock)
    return n.outputs[2]
def clay():
    m, nt, b = mat("clay"); b.inputs["Base Color"].default_value = (0.62, 0.62, 0.62, 1); b.inputs["Roughness"].default_value = 0.9; return m
# ---------------- materials ----------------
def m_ground():
    m, nt, b = mat("ground"); img = json.load(open(f"{root}/data/imagery.json"))
    xyz = N(nt, "ShaderNodeSeparateXYZ"); nt.links.new(N(nt, "ShaderNodeNewGeometry").outputs["Position"], xyz.inputs[0]); col = None
    for key in ("far", "city", "near"):
        e = img[key]["extent"]; u = M(nt, "DIVIDE", M(nt, "SUBTRACT", xyz.outputs[0], e[0]), e[2] - e[0]); v = M(nt, "DIVIDE", M(nt, "SUBTRACT", xyz.outputs[1], e[1]), e[3] - e[1])
        cmb = N(nt, "ShaderNodeCombineXYZ"); nt.links.new(u, cmb.inputs[0]); nt.links.new(v, cmb.inputs[1])
        t = N(nt, "ShaderNodeTexImage", extension="EXTEND"); t.image = bpy.data.images.load(f"{root}/{img[key]['file']}"); nt.links.new(cmb.outputs[0], t.inputs[0])
        if col is None: col = t.outputs[0]
        else:
            k = 0.01 if key == "near" else 0.002
            col = mix(nt, M(nt, "MULTIPLY", band(nt, u, k, 1 - k), band(nt, v, k, 1 - k)), col, t.outputs[0])
    nt.links.new(col, b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.85; return m
def m_wall(windows=True):
    m, nt, b = mat("wall" if windows else "wall_plain"); col = N(nt, "ShaderNodeAttribute", attribute_name="Col").outputs[0]
    noise = N(nt, "ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 0.08
    base = mix(nt, 0.25, col, mix(nt, noise.outputs[0], (0.55, 0.53, 0.5), (1, 1, 1)))
    if windows:
        uv = N(nt, "ShaderNodeSeparateXYZ"); nt.links.new(N(nt, "ShaderNodeUVMap").outputs[0], uv.inputs[0]); u, v = uv.outputs[0], uv.outputs[1]
        BAY, FL, G = 3.4, 3.1, 3.6
        vf = M(nt, "SUBTRACT", v, G); fu = M(nt, "FRACT", M(nt, "DIVIDE", u, BAY)); fv = M(nt, "FRACT", M(nt, "DIVIDE", vf, FL))
        upper = M(nt, "GREATER_THAN", vf, 0.0)
        win = M(nt, "MULTIPLY", upper, M(nt, "MULTIPLY", band(nt, fu, 0.16, 0.84), band(nt, fv, 0.24, 0.84)))
        slab = M(nt, "MULTIPLY", upper, M(nt, "LESS_THAN", fv, 0.07))                       # balcony slab line
        rail = M(nt, "MULTIPLY", upper, M(nt, "MULTIPLY", band(nt, fv, 0.07, 0.36), band(nt, fu, 0.08, 0.92)))
        shop = M(nt, "MULTIPLY", band(nt, v, 0.5, 3.0), band(nt, M(nt, "FRACT", M(nt, "DIVIDE", u, 4.2)), 0.1, 0.9))
        cell = N(nt, "ShaderNodeCombineXYZ"); nt.links.new(M(nt, "FLOOR", M(nt, "DIVIDE", u, BAY)), cell.inputs[0]); nt.links.new(M(nt, "FLOOR", M(nt, "DIVIDE", vf, FL)), cell.inputs[1])
        wn = N(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D"); nt.links.new(cell.outputs[0], wn.inputs["Vector"])
        glass = mix(nt, wn.outputs["Value"], (0.035, 0.045, 0.06), (0.16, 0.17, 0.18))
        awn = mix(nt, M(nt, "GREATER_THAN", wn.outputs["Value"], 0.78), glass, (0.50, 0.47, 0.40))   # some shutters are down
        c = mix(nt, slab, base, mix(nt, 0.5, base, (0.25, 0.25, 0.25)))
        c = mix(nt, win, c, awn); c = mix(nt, shop, c, (0.05, 0.06, 0.07))
        mpd = N(nt, "ShaderNodeMapping"); mpd.inputs["Scale"].default_value = (0.9, 0.06, 1); nt.links.new(N(nt, "ShaderNodeUVMap").outputs[0], mpd.inputs[0])
        dirt = N(nt, "ShaderNodeTexNoise"); dirt.inputs["Scale"].default_value = 1.0; dirt.inputs["Detail"].default_value = 5; nt.links.new(mpd.outputs[0], dirt.inputs["Vector"])
        hole = M(nt, "MAXIMUM", win, shop)
        c = mix(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "SUBTRACT", dirt.outputs[0], 0.42), 1.1), M(nt, "SUBTRACT", 1.0, hole)), c, (0.30, 0.28, 0.25))   # rain dirt
        nt.links.new(c, b.inputs["Base Color"])
        nt.links.new(M(nt, "SUBTRACT", 0.8, M(nt, "MULTIPLY", hole, 0.74)), b.inputs["Roughness"])
        fine = N(nt, "ShaderNodeTexNoise"); fine.inputs["Scale"].default_value = 6.0; nt.links.new(N(nt, "ShaderNodeUVMap").outputs[0], fine.inputs["Vector"])
        bp = N(nt, "ShaderNodeBump"); bp.inputs["Strength"].default_value = 1.0; bp.inputs["Distance"].default_value = 0.25
        nt.links.new(M(nt, "ADD", M(nt, "MULTIPLY", hole, -1.0), M(nt, "MULTIPLY", M(nt, "MULTIPLY", fine.outputs[0], 0.03), M(nt, "SUBTRACT", 1.0, hole))), bp.inputs["Height"]); nt.links.new(bp.outputs[0], b.inputs["Normal"])
    else:
        nt.links.new(base, b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.85
    return m
def m_roof():
    m, nt, b = mat("roof"); col = N(nt, "ShaderNodeAttribute", attribute_name="Col").outputs[0]
    n1 = N(nt, "ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 0.35; n1.inputs["Detail"].default_value = 6
    nt.links.new(mix(nt, 0.45, col, mix(nt, n1.outputs[0], (0.25, 0.25, 0.25), (0.95, 0.93, 0.9))), b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.9; return m
def m_plain(name, rgb, rough=0.7, metal=0.0, attr=False):
    m, nt, b = mat(name); b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    if attr: nt.links.new(N(nt, "ShaderNodeAttribute", attribute_name="Col").outputs[0], b.inputs["Base Color"])
    return m
def m_stone():
    """photographed rubble masonry, lightened towards the tower's pale buff, with stains, streaks and grime on top."""
    m, nt, b = mat("tower_stone"); uvn = N(nt, "ShaderNodeUVMap"); c = pbr(nt, b, "stone", uvn.outputs[0], 0.36, bump=1.6)
    c = mul(nt, c, (1.0, 0.97, 0.90)); c = mix(nt, 0.5, c, (0.86, 0.80, 0.69))
    big = N(nt, "ShaderNodeTexNoise"); big.inputs["Scale"].default_value = 0.16; big.inputs["Detail"].default_value = 8; big.inputs["Roughness"].default_value = 0.65
    nt.links.new(N(nt, "ShaderNodeNewGeometry").outputs["Position"], big.inputs["Vector"])
    c = mix(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", big.outputs[0], 0.42), 1.2), c, (0.86, 0.82, 0.74))        # pale lime-washed patches
    c = mix(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", 0.5, big.outputs[0]), 0.9), c, (0.44, 0.38, 0.30))          # darker weathered patches
    uvs = N(nt, "ShaderNodeSeparateXYZ"); nt.links.new(uvn.outputs[0], uvs.inputs[0])
    mp2 = N(nt, "ShaderNodeMapping"); mp2.inputs["Scale"].default_value = (1.4, 0.05, 1); nt.links.new(uvn.outputs[0], mp2.inputs[0])
    st = N(nt, "ShaderNodeTexNoise"); st.inputs["Scale"].default_value = 1.0; st.inputs["Detail"].default_value = 4; nt.links.new(mp2.outputs[0], st.inputs["Vector"])
    c = mix(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", st.outputs[0], 0.5), 1.0), c, (0.22, 0.20, 0.17))           # rain streaks
    low = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, M(nt, "MINIMUM", M(nt, "DIVIDE", uvs.outputs[1], 6.0), 1.0)), 0.3)
    c = mix(nt, low, c, (0.22, 0.19, 0.16)); nt.links.new(c, b.inputs["Base Color"]); return m
def m_brick():
    m, nt, b = mat("tower_brick"); uvn = N(nt, "ShaderNodeUVMap"); c = pbr(nt, b, "brick", uvn.outputs[0], 0.5, bump=1.2)
    nz = N(nt, "ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.7; nz.inputs["Detail"].default_value = 6; nt.links.new(uvn.outputs[0], nz.inputs["Vector"])
    c = mul(nt, c, (1.25, 1.0, 0.85)); c = mix(nt, 0.25, c, (0.72, 0.36, 0.20)); nt.links.new(mix(nt, M(nt, "MULTIPLY", nz.outputs[0], 0.3), c, (0.56, 0.44, 0.30)), b.inputs["Base Color"]); return m
def m_cap():
    m, nt, b = mat("tower_cap"); nz = N(nt, "ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.6; nz.inputs["Detail"].default_value = 7
    nt.links.new(N(nt, "ShaderNodeNewGeometry").outputs["Position"], nz.inputs["Vector"])
    nt.links.new(mix(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs[0], 0.42), 2.6), (0.60, 0.55, 0.46), (0.62, 0.50, 0.13)), b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.9; return m
def m_site(name, kind, site):
    """surfaces around the tower; they fade out to the satellite photo at the edge of the site."""
    m, nt, b = mat(name); pos = N(nt, "ShaderNodeNewGeometry").outputs["Position"]; b.inputs["Roughness"].default_value = 0.9
    def noise(scale, detail=4):
        n = N(nt, "ShaderNodeTexNoise"); n.inputs["Scale"].default_value = scale; n.inputs["Detail"].default_value = detail; nt.links.new(pos, n.inputs["Vector"]); return n.outputs[0]
    def bricks(w, h, c1, c2, mortar, msize=0.012):
        br = N(nt, "ShaderNodeTexBrick", offset=0.5); nt.links.new(pos, br.inputs["Vector"])
        for k, val in (("Scale", 1.0), ("Mortar Size", msize), ("Bias", 0.0), ("Brick Width", w), ("Row Height", h)): br.inputs[k].default_value = val
        br.inputs["Color1"].default_value = (*c1, 1); br.inputs["Color2"].default_value = (*c2, 1); br.inputs["Mortar"].default_value = (*mortar, 1); return br.outputs["Color"]
    if kind == "grass":
        c = pbr(nt, b, "grass", pos, 0.28, bump=0.6); c = mix(nt, 0.55, c, mix(nt, noise(0.08, 3), (0.10, 0.24, 0.05), (0.22, 0.36, 0.09)))
        c = mix(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", noise(0.05), 0.55), 1.0), c, (0.36, 0.33, 0.16))
    elif kind == "pave": c = pbr(nt, b, "prom", pos, 0.3); c = mix(nt, 0.45, c, mix(nt, noise(0.06, 4), (0.56, 0.44, 0.33), (0.70, 0.60, 0.48)))
    elif kind == "prom": c = pbr(nt, b, "prom", pos, 0.22); c = mix(nt, 0.35, c, mix(nt, noise(0.05, 4), (0.58, 0.54, 0.47), (0.74, 0.70, 0.62)))
    elif kind == "cobble": c = pbr(nt, b, "cobble", pos, 0.45, bump=1.5)
    elif kind == "asphalt": c = pbr(nt, b, "asphalt", pos, 0.2); c = mul(nt, c, mix(nt, noise(0.1, 4), (0.55, 0.55, 0.57), (0.95, 0.95, 0.95)))
    elif kind == "walk": c = pbr(nt, b, "prom", pos, 0.5); c = mix(nt, 0.3, c, (0.60, 0.60, 0.58))
    elif kind == "mark": c = (0.82, 0.82, 0.78)
    elif kind == "terra": c = pbr(nt, b, "paving", pos, 0.4); c = mix(nt, 0.4, c, mix(nt, noise(0.06, 4), (0.50, 0.33, 0.24), (0.62, 0.45, 0.34)))
    elif kind == "dirt": c = mix(nt, noise(0.4, 5), (0.50, 0.42, 0.30), (0.66, 0.57, 0.42))
    else: c = (0.58, 0.55, 0.50)
    if isinstance(c, tuple): b.inputs["Base Color"].default_value = (*c, 1)
    else: nt.links.new(c, b.inputs["Base Color"])
    sep = N(nt, "ShaderNodeSeparateXYZ"); nt.links.new(pos, sep.inputs[0])
    dx = M(nt, "SUBTRACT", sep.outputs[0], site["x"]); dy = M(nt, "SUBTRACT", sep.outputs[1], site["y"])
    d = M(nt, "SQRT", M(nt, "ADD", M(nt, "MULTIPLY", dx, dx), M(nt, "MULTIPLY", dy, dy)))
    fade = N(nt, "ShaderNodeMapRange"); fade.inputs[1].default_value = 205; fade.inputs[2].default_value = 255; nt.links.new(d, fade.inputs[0])
    ms = N(nt, "ShaderNodeMixShader"); tr = N(nt, "ShaderNodeBsdfTransparent"); out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    nt.links.new(fade.outputs[0], ms.inputs[0]); nt.links.new(b.outputs[0], ms.inputs[1]); nt.links.new(tr.outputs[0], ms.inputs[2]); nt.links.new(ms.outputs[0], out.inputs[0]); return m
def m_glow():
    m, nt, b = mat("lamp_glass"); b.inputs["Base Color"].default_value = (0.9, 0.88, 0.8, 1); b.inputs["Emission Color"].default_value = (1, 0.9, 0.7, 1); b.inputs["Emission Strength"].default_value = 0.6; return m
def m_leaf(name):
    """leaves: colour per leaf, a little light passes through."""
    m, nt, b = mat(name); col = N(nt, "ShaderNodeAttribute", attribute_name="Col").outputs[0]; nt.links.new(col, b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.6
    tr = N(nt, "ShaderNodeBsdfTranslucent"); nt.links.new(mul(nt, col, (1.6, 1.9, 0.7)), tr.inputs["Color"]); ms = N(nt, "ShaderNodeMixShader"); ms.inputs[0].default_value = 0.3
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"); nt.links.new(b.outputs[0], ms.inputs[1]); nt.links.new(tr.outputs[0], ms.inputs[2]); nt.links.new(ms.outputs[0], out.inputs[0]); return m
def m_water():
    """sea: dark green-blue, mirror-like, ruffled by two scales of waves."""
    m, nt, b = mat("sea"); pos = N(nt, "ShaderNodeNewGeometry").outputs["Position"]
    big = N(nt, "ShaderNodeTexNoise"); big.inputs["Scale"].default_value = 0.004; big.inputs["Detail"].default_value = 3; nt.links.new(pos, big.inputs["Vector"])
    nt.links.new(mix(nt, big.outputs[0], (0.012, 0.085, 0.13), (0.02, 0.14, 0.19)), b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.06
    try: b.inputs["Specular IOR Level"].default_value = 0.6
    except KeyError: pass
    h = None
    for sc_, st in ((0.9, 0.5), (0.22, 0.9), (0.05, 1.2)):
        mp = N(nt, "ShaderNodeMapping"); mp.inputs["Scale"].default_value = (sc_, sc_ * 1.8, sc_); mp.inputs["Rotation"].default_value = (0, 0, 0.6 + sc_); nt.links.new(pos, mp.inputs[0])
        n = N(nt, "ShaderNodeTexNoise"); n.inputs["Scale"].default_value = 1.0; n.inputs["Detail"].default_value = 5; n.inputs["Roughness"].default_value = 0.6; nt.links.new(mp.outputs[0], n.inputs["Vector"])
        v = M(nt, "MULTIPLY", n.outputs[0], st); h = v if h is None else M(nt, "ADD", h, v)
    bp = N(nt, "ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.6; bp.inputs["Distance"].default_value = 0.8; nt.links.new(h, bp.inputs["Height"]); nt.links.new(bp.outputs[0], b.inputs["Normal"]); return m
# ---------------- scene ----------------
site = json.load(open(f"{root}/data/tower_site.json")); TP = Vector((site["x"], site["y"], site["ground"] - 0.15))
C = clay() if CLAY else None
def setm(ob, mats):
    for mm in mats: ob.data.materials.append(C or mm)
if not ONLY_TOWER:
    setm(load("ground"), [m_ground()])
    setm(load("quay"), [m_plain("quay", (0.45, 0.44, 0.42), 0.9)])
    setm(load("buildings"), [m_wall(True), m_roof(), m_plain("plain", (0.8, 0.8, 0.8), 0.8, attr=True), m_plain("glass", (0.05, 0.06, 0.08), 0.12, attr=True)])
    setm(load("trees", smooth=False), [m_leaf("tree")])
    bpy.ops.mesh.primitive_plane_add(size=90000); sea = bpy.context.object; sea.name = "sea"; sea.location = (0, 0, 0.35); setm(sea, [m_water()])
    setm(load("site", smooth=False), [m_site("grass", "grass", site), m_site("paving", "pave", site), m_site("cobble", "cobble", site), m_site("asphalt", "asphalt", site),
         m_site("kerb", "kerb", site), m_site("promenade", "prom", site), m_plain("lamp", (0.03, 0.07, 0.05), 0.5, 0.6), m_glow(), m_plain("car", (0.5, 0.5, 0.5), 0.22, 0.5, attr=True),
         m_plain("car_glass", (0.02, 0.025, 0.03), 0.1), m_plain("bark", (0.19, 0.13, 0.09), 0.9), m_leaf("pine"), m_site("marking", "mark", site), m_site("sidewalk", "walk", site), m_site("terracotta", "terra", site), m_plain("marble", (0.80, 0.80, 0.78), 0.5), m_site("dirt", "dirt", site)])
else:
    bpy.ops.mesh.primitive_plane_add(size=600); bpy.context.object.location = TP; setm(bpy.context.object, [m_plain("floor", (0.5, 0.5, 0.5), 0.9)])
tw = load("tower"); tw.location = TP
setm(tw, [m_stone(), m_plain("dark", (0.012, 0.012, 0.014), 0.6), m_plain("pave", (0.50, 0.46, 0.40), 0.9), m_plain("flag_white", (0.9, 0.9, 0.9), 0.7),
          m_plain("flag_blue", (0.02, 0.16, 0.55), 0.7), m_plain("metal", (0.12, 0.12, 0.12), 0.5, 1.0), m_brick(), m_cap()])
# light: a photographed clear sky (Poly Haven), turned so that its sun sits at the azimuth we want
SUN_AZ = math.radians(float(arg("--sun-az", "225"))); HDRI_SUN_MATH = math.radians(171.4)        # where the sun is in the file
w = bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True; wn = w.node_tree; bg = wn.nodes["Background"]
env = wn.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(f"{root}/{ASSETS['sky']}")
mpw = wn.nodes.new("ShaderNodeMapping"); mpw.vector_type = "POINT"; tcw = wn.nodes.new("ShaderNodeTexCoord")
want = math.pi / 2 - SUN_AZ; mpw.inputs["Rotation"].default_value = (0, 0, float(arg("--sky-sign", "1")) * (want - HDRI_SUN_MATH))
wn.links.new(tcw.outputs["Generated"], mpw.inputs[0]); wn.links.new(mpw.outputs[0], env.inputs[0]); wn.links.new(env.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = float(arg("--sky", "1.0"))
skytype = "HDRI"; sc.view_layers[0].use_pass_mist = True; w.mist_settings.start = 0; w.mist_settings.depth = 30000; w.mist_settings.falloff = "LINEAR"
# cameras
CAMDEF = {"aerial": ((-430, -360, 250), (40, 60, 5), 35), "tower": ((-66, -58, 24), (0, 0, 16), 32), "city": ((-1500, -1700, 1150), (150, 250, 0), 32),
          "tower_ne": ((62, 48, 30), (0, 0, 16), 32), "ref5": ((92, 70, 40), (2, 0, 17), 50), "ref1": ((27, 31, 1.7), (0, 0, 19), 22), "crown": ((-30, -18, 37), (0, 0, 27), 35), "earth": ((-185, -75, 82), (8, 12, 6), 32), "park": ((-78, -96, 58), (6, -10, 2), 30), "strip": ((-40, -150, 70), (20, -60, 0), 30), "planters": ((-34, -108, 1.7), (-2, -20, 12), 24), "drone": ((95, -150, 62), (-12, 25, 10), 26), "plaza": ((-50, -34, 1.7), (0, 0, 15), 20), "theatre": ((35, -22, 26), (118, 30, 11), 28), "nikis": ((-150, 20, 45), (-40, 150, 12), 30), "suntest": ((0, -1, 170), (0, 0, 0), 28), "top": ((0, -1, 2600), (0, 0, 0), 28), "waterfront": ((-260, 150, 60), (-10, -10, 12), 40)}
for name, (pos, tgt, lens) in CAMDEF.items():
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_end = 80000; cd.clip_start = 0.5
    cam = bpy.data.objects.new("cam_" + name, cd); sc.collection.objects.link(cam)
    p = Vector(pos) + Vector((TP.x, TP.y, TP.z)); tg = Vector(tgt) + Vector((TP.x, TP.y, TP.z)); cam.location = p; cam.rotation_euler = (tg - p).to_track_quat("-Z", "Y").to_euler()
# render settings
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True; sc.cycles.max_bounces = 4
sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.03; sc.cycles.sample_clamp_indirect = 8
sc.render.resolution_x = RESX; sc.render.resolution_y = int(RESX * 9 / 16); sc.render.threads_mode = "AUTO"
try: sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
except TypeError: pass
sc.view_settings.exposure = float(arg("--exposure", "0")); sc.render.image_settings.file_format = "PNG"
if "--save" in argv:
    if "--pack" in argv:                                   # one self-contained, lighter file to take away
        for name in ("trees", "ground"):
            me = bpy.data.objects[name].data
            while me.uv_layers: me.uv_layers.remove(me.uv_layers[0])
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=f"{root}/out/{arg('--blend', 'thessaloniki')}.blend", compress="--pack" in argv)
print("SKY", skytype, "objects", len(sc.objects))
import OpenImageIO as oiio
for name in CAMS:
    sc.camera = bpy.data.objects["cam_" + name]; base = f"{root}/out/{arg('--prefix', '')}{name}{'_clay' if CLAY else ''}"
    sc.render.image_settings.media_type = "MULTI_LAYER_IMAGE"; sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"; sc.render.filepath = base + ".exr"
    bpy.ops.render.render(write_still=True)
    sc.render.image_settings.media_type = "IMAGE"; sc.render.image_settings.file_format = "PNG"; bpy.data.images["Render Result"].save_render(base + "_raw.png", scene=sc)
    mist = None; sub = 0
    while mist is None:
        inp = oiio.ImageInput.open(base + ".exr", oiio.ImageSpec()) if sub == 0 else inp
        if not inp.seek_subimage(sub, 0): break
        spec = inp.spec(); ch = [i for i, c in enumerate(spec.channelnames) if "mist" in c.lower()]
        if ch: mist = inp.read_image(sub, 0, ch[0], ch[0] + 1, "float")
        else: print("SUB", sub, spec.channelnames)
        sub += 1
    inp.close(); np.save(base + "_mist.npy", np.asarray(mist, np.float32).reshape(spec.height, spec.width)); os.remove(base + ".exr")
    print("RENDER", name)
