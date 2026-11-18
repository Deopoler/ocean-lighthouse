"""Lighthouse on a rocky islet -- fully procedural Blender 5.x scene builder.

Run:  blender --python scripts/lighthouse.py     (or exec() inside an open scene)

Builds everything inside the collection "Lighthouse" (rebuilt idempotently):
  * LH_Islet (+ LH_Boulders)      rocky mound ~30 x 22 m at (10, 80), top ~6 m, bottom -6.5 m
  * LH_Tower & friends            tapered masonry tower, gallery, lantern room, dome
  * LH_Keeper*                    keeper's house (stone walls, slate roof, chimney)
  * LH_BeamPivot (empty, driver)  rotating beams (2 additive cones) + 2 spot lights + lamp light
Only bpy / bmesh / mathutils are used, no external files.
"""
import math
import random

import bpy
import bmesh
from mathutils import Vector, Matrix
import mathutils.noise as mnoise

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
COLL_NAME = "Lighthouse"
PFX = "LH_"

ISLET_C = Vector((10.0, 80.0, 0.0))     # islet centre (world)
ISLET_RX, ISLET_RY = 15.0, 11.0         # waterline semi-axes (30 x 22 m)
PLATEAU = 5.8                           # plateau height
BOTTOM = -6.5                           # underside of the islet
PAD = 5.8                               # flat building pad height (islet-local z == world z)

# building positions, relative to islet centre
TOWER_REL = Vector((-2.6, 1.4))
HOUSE_REL = Vector((4.6, -2.0))
HOUSE_W, HOUSE_D, HOUSE_H, HOUSE_ROT = 5.8, 3.8, 2.8, 0.10
TOWER_ORG = Vector((ISLET_C.x + TOWER_REL.x, ISLET_C.y + TOWER_REL.y, PAD))
HOUSE_ORG = Vector((ISLET_C.x + HOUSE_REL.x, ISLET_C.y + HOUSE_REL.y, PAD))

# tower metrics (tower-local z, origin = base on the pad)
Z_SH0, Z_SH1 = 0.92, 17.4               # shaft start / end
R_SH0, R_SH1 = 3.05, 2.15               # shaft radius bottom / top
Z_GAL = 18.55                           # gallery floor
Z_LAMP = 20.4                           # lamp centre
DOOR_AZ = math.radians(-100.0)          # door faces the camera approach

BEAM_LEN = 150.0
BEAM_PERIOD = 144.0

SEG_TOWER = 72


# --------------------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else (b if x > b else x)


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


def Rz(a):
    return Matrix.Rotation(a, 4, 'Z')


def Tr(x, y=0.0, z=0.0):
    if isinstance(x, Vector):
        return Matrix.Translation(x)
    return Matrix.Translation((x, y, z))


def blackbody_rgb(kelvin):
    """Approximate linear RGB (max-normalised) of a black body."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307

    def lin(c):
        c = clamp(c / 255.0)
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rgb = [lin(r), lin(g), lin(b)]
    m = max(rgb)
    return tuple(c / m for c in rgb)


# --------------------------------------------------------------------------------------
# Cleanup (idempotency)
# --------------------------------------------------------------------------------------
def cleanup():
    coll = bpy.data.collections.get(COLL_NAME)
    meshes, lights, mats = [], [], []
    if coll is not None:
        for ob in list(coll.all_objects):
            d = ob.data
            if d is not None:
                if isinstance(d, bpy.types.Mesh):
                    meshes.append(d)
                    mats.extend([m for m in d.materials if m])
                elif isinstance(d, bpy.types.Light):
                    lights.append(d)
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(coll)
    for me in meshes:
        if me.name in bpy.data.meshes and me.users == 0:
            bpy.data.meshes.remove(me)
    for li in lights:
        if li.name in bpy.data.lights and li.users == 0:
            bpy.data.lights.remove(li)
    # stray leftovers of earlier runs (only our own prefix)
    for me in list(bpy.data.meshes):
        if me.name.startswith(PFX) and me.users == 0:
            bpy.data.meshes.remove(me)
    for li in list(bpy.data.lights):
        if li.name.startswith(PFX) and li.users == 0:
            bpy.data.lights.remove(li)
    for m in set(mats) | {m for m in bpy.data.materials if m.name.startswith(PFX)}:
        if m.name in bpy.data.materials and m.users == 0:
            bpy.data.materials.remove(m)
    for a in list(bpy.data.actions):
        if a.name.startswith(PFX) and a.users == 0:
            bpy.data.actions.remove(a)


# --------------------------------------------------------------------------------------
# Shader-node helper
# --------------------------------------------------------------------------------------
class Sh:
    """Tiny helper to build node trees tersely.  Values may be constants or sockets."""

    def __init__(self, name):
        self.mat = bpy.data.materials.new(name)
        if self.mat.node_tree is None:
            self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new('ShaderNodeOutputMaterial')
        self.out.location = (2400, 0)
        self._n = 0

    # -- basics ------------------------------------------------------------------------
    def new(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = ((self._n % 9) * 260 - 200, -(self._n // 9) * 300)
        self._n += 1
        return n

    def set(self, node, key, val):
        if val is None:
            return
        sock = node.inputs[key]
        if isinstance(val, bpy.types.NodeSocket):
            self.nt.links.new(val, sock)
            return
        dv = sock.default_value
        if hasattr(dv, '__len__'):
            n = len(dv)
            if not hasattr(val, '__len__'):
                val = [val] * n
            val = list(val)
            while len(val) < n:
                val.append(1.0)
            val = val[:n]
        sock.default_value = val

    # -- coordinates / vector ----------------------------------------------------------
    def tex_coord(self):
        return self.new('ShaderNodeTexCoord')

    def geometry(self):
        return self.new('ShaderNodeNewGeometry')

    def sepxyz(self, vec):
        n = self.new('ShaderNodeSeparateXYZ')
        self.set(n, 'Vector', vec)
        return n.outputs['X'], n.outputs['Y'], n.outputs['Z']

    def comb(self, x=0.0, y=0.0, z=0.0):
        n = self.new('ShaderNodeCombineXYZ')
        self.set(n, 'X', x)
        self.set(n, 'Y', y)
        self.set(n, 'Z', z)
        return n.outputs[0]

    def mapping(self, vec, scale=(1, 1, 1), loc=(0, 0, 0)):
        n = self.new('ShaderNodeMapping', vector_type='POINT')
        self.set(n, 'Vector', vec)
        self.set(n, 'Scale', scale)
        self.set(n, 'Location', loc)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.new('ShaderNodeVectorMath', operation=op)
        self.set(n, 0, a)
        if b is not None:
            self.set(n, 1, b)
        if scale is not None:
            self.set(n, 3, scale)
        return n

    def vadd(self, a, b):
        return self.vmath('ADD', a, b).outputs[0]

    def vsub(self, a, b):
        return self.vmath('SUBTRACT', a, b).outputs[0]

    def vscale(self, a, s):
        return self.vmath('SCALE', a, None, s).outputs[0]

    def dot(self, a, b):
        return self.vmath('DOT_PRODUCT', a, b).outputs['Value']

    # -- scalar -------------------------------------------------------------------------
    def math(self, op, a, b=None, c=None, clamp_=False):
        n = self.new('ShaderNodeMath', operation=op, use_clamp=clamp_)
        self.set(n, 0, a)
        if b is not None:
            self.set(n, 1, b)
        if c is not None:
            self.set(n, 2, c)
        return n.outputs[0]

    def add(self, a, b):
        return self.math('ADD', a, b)

    def mul(self, a, b):
        return self.math('MULTIPLY', a, b)

    def sub(self, a, b):
        return self.math('SUBTRACT', a, b)

    def inv(self, a):
        return self.math('SUBTRACT', 1.0, a)

    def mx(self, a, b):
        return self.math('MAXIMUM', a, b)

    def mn(self, a, b):
        return self.math('MINIMUM', a, b)

    def pw(self, a, p):
        return self.math('POWER', a, p)

    def mr(self, v, a, b, c=0.0, d=1.0, interp='LINEAR', clamp_=True):
        n = self.new('ShaderNodeMapRange', data_type='FLOAT', interpolation_type=interp, clamp=clamp_)
        self.set(n, 0, v)
        self.set(n, 1, a)
        self.set(n, 2, b)
        self.set(n, 3, c)
        self.set(n, 4, d)
        return n.outputs[0]

    def mixf(self, fac, a, b):
        n = self.new('ShaderNodeMix', data_type='FLOAT', clamp_factor=True)
        self.set(n, 0, fac)
        self.set(n, 2, a)
        self.set(n, 3, b)
        return n.outputs[0]

    def mixc(self, fac, a, b, blend='MIX'):
        n = self.new('ShaderNodeMix', data_type='RGBA', blend_type=blend, clamp_factor=True)
        self.set(n, 0, fac)
        self.set(n, 6, a)
        self.set(n, 7, b)
        return n.outputs[2]

    def ramp(self, fac, stops, interp='LINEAR'):
        n = self.new('ShaderNodeValToRGB')
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        while len(cr.elements) > len(stops):
            cr.elements.remove(cr.elements[len(cr.elements) - 1])
        # positions first (ascending), then colours
        for e, (p, c) in zip(cr.elements, stops):
            e.position = p
        for e, (p, c) in zip(cr.elements, stops):
            c = list(c) if hasattr(c, '__len__') else [c, c, c]
            while len(c) < 4:
                c.append(1.0)
            e.color = c
        self.set(n, 'Fac', fac)
        return n.outputs['Color']

    def gray(self, v):
        n = self.new('ShaderNodeRGBToBW')
        self.set(n, 'Color', v)
        return n.outputs[0]

    # -- textures -----------------------------------------------------------------------
    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0, lac=2.0):
        n = self.new('ShaderNodeTexNoise', noise_dimensions='3D')
        self.set(n, 'Vector', vec)
        self.set(n, 'Scale', scale)
        self.set(n, 'Detail', detail)
        self.set(n, 'Roughness', rough)
        self.set(n, 'Distortion', distortion)
        self.set(n, 'Lacunarity', lac)
        return n

    def voronoi(self, vec, scale, feature='F1', rand=1.0, smooth=None):
        n = self.new('ShaderNodeTexVoronoi', voronoi_dimensions='3D', feature=feature)
        self.set(n, 'Vector', vec)
        self.set(n, 'Scale', scale)
        self.set(n, 'Randomness', rand)
        if smooth is not None:
            self.set(n, 'Smoothness', smooth)
        return n

    def brick(self, vec, scale, width, height, mortar, smooth, c1, c2, cm, bias=0.0, offset=0.5,
              offset_freq=2, squash=1.0, squash_freq=2):
        n = self.new('ShaderNodeTexBrick', offset=offset, offset_frequency=offset_freq, squash=squash,
                     squash_frequency=squash_freq)
        self.set(n, 'Vector', vec)
        self.set(n, 'Scale', scale)
        self.set(n, 'Brick Width', width)
        self.set(n, 'Row Height', height)
        self.set(n, 'Mortar Size', mortar)
        self.set(n, 'Mortar Smooth', smooth)
        self.set(n, 'Bias', bias)
        self.set(n, 'Color1', c1)
        self.set(n, 'Color2', c2)
        self.set(n, 'Mortar', cm)
        return n

    def bump(self, height, strength=1.0, dist=0.1, normal=None):
        n = self.new('ShaderNodeBump')
        self.set(n, 'Height', height)
        self.set(n, 'Strength', strength)
        self.set(n, 'Distance', dist)
        if normal is not None:
            self.set(n, 'Normal', normal)
        return n.outputs['Normal']

    def displacement(self, height, mid, scale):
        n = self.new('ShaderNodeDisplacement')
        self.set(n, 'Height', height)
        self.set(n, 'Midlevel', mid)
        self.set(n, 'Scale', scale)
        self.nt.links.new(n.outputs['Displacement'], self.out.inputs['Displacement'])

    # -- shaders ------------------------------------------------------------------------
    def principled(self, **kv):
        """kv keys use '_' for spaces."""
        n = self.new('ShaderNodeBsdfPrincipled')
        for k, v in kv.items():
            self.set(n, k.replace('_', ' '), v)
        return n.outputs['BSDF']

    def finish(self, shader):
        self.nt.links.new(shader, self.out.inputs['Surface'])
        return self.mat


# --------------------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------------------
def mat_rock():
    s = Sh(PFX + "Rock")
    tc = s.tex_coord()
    P = tc.outputs['Object']
    geo = s.geometry()
    wz = s.sepxyz(geo.outputs['Position'])[2]
    nz = s.sepxyz(geo.outputs['Normal'])[2]

    # domain warped coordinates -> natural looking cracks
    warp = s.noise(P, 0.22, 3, 0.55).outputs['Color']
    wp = s.vadd(P, s.vscale(s.vsub(warp, (0.5, 0.5, 0.5)), 5.0))

    # cracks (three scales)
    gate = s.mr(s.noise(P, 0.5, 2, 0.5).outputs['Fac'], 0.35, 0.55)
    e1 = s.voronoi(wp, 0.30, 'DISTANCE_TO_EDGE').outputs['Distance']
    e2 = s.voronoi(wp, 1.15, 'DISTANCE_TO_EDGE').outputs['Distance']
    e3 = s.voronoi(wp, 4.5, 'DISTANCE_TO_EDGE').outputs['Distance']
    c1 = s.mul(s.mr(e1, 0.0, 0.05, 1.0, 0.0), gate)
    c2 = s.mul(s.mr(e2, 0.0, 0.06, 1.0, 0.0), 0.85)
    c3 = s.mul(s.mr(e3, 0.0, 0.07, 1.0, 0.0), 0.5)
    crack = s.mx(c1, s.mx(c2, c3))

    # relief
    n_big = s.noise(wp, 2.2, 10, 0.6).outputs['Fac']
    n_fine = s.noise(P, 28.0, 6, 0.6).outputs['Fac']
    nod = s.voronoi(wp, 1.6, 'F1').outputs['Distance']
    block = s.voronoi(wp, 0.30, 'F1')
    height = s.add(s.add(s.mul(n_big, 0.55), s.mul(s.inv(nod), 0.25)), s.mul(n_fine, 0.20))
    height = s.sub(height, s.mul(crack, 0.75))

    # colour: dark granite / basalt
    cn = s.noise(wp, 1.4, 6, 0.55).outputs['Fac']
    col = s.ramp(cn, [(0.30, (0.030, 0.031, 0.035)), (0.55, (0.075, 0.072, 0.072)), (0.78, (0.135, 0.128, 0.120))])
    blk_tint = s.mr(s.gray(block.outputs['Color']), 0.1, 0.9, 0.65, 1.3)
    tint_n = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(tint_n, 0, 1.0)
    s.set(tint_n, 6, col)
    s.set(tint_n, 7, s.comb(blk_tint, blk_tint, blk_tint))
    col = tint_n.outputs[2]
    # granite flecks
    fl_d = s.voronoi(P, 70.0, 'F1').outputs['Distance']
    fl = s.mul(s.mr(fl_d, 0.35, 0.05, 0.0, 1.0), s.mr(s.noise(P, 4.0, 2, 0.5).outputs['Fac'], 0.45, 0.62))
    col = s.mixc(s.mul(fl, 0.55), col, (0.30, 0.285, 0.27, 1.0))
    # rust / iron staining
    stain = s.mul(s.mr(s.noise(wp, 1.1, 4, 0.55).outputs['Fac'], 0.55, 0.72), 0.18)
    col = s.mixc(stain, col, (0.11, 0.065, 0.04, 1.0))
    col = s.mixc(s.mul(crack, 0.8), col, (0.012, 0.012, 0.014, 1.0))

    # lichen (crust patches) on upward facing, above the splash zone
    gate_l = s.mr(s.noise(P, 0.9, 5, 0.6).outputs['Fac'], 0.42, 0.6)
    top_l = s.mr(s.add(nz, s.mul(s.sub(gate_l, 0.5), 0.3)), 0.52, 0.86)
    alt = s.mr(wz, 1.8, 3.4)
    patch = s.mr(s.voronoi(wp, 2.4, 'F1').outputs['Distance'], 0.62, 0.30)
    lich = s.mul(s.mul(top_l, alt), s.mul(patch, gate_l))
    lich_col = s.mixc(s.mr(s.noise(P, 3.0, 3, 0.5).outputs['Fac'], 0.56, 0.68),
                      (0.34, 0.37, 0.25, 1.0), (0.55, 0.28, 0.05, 1.0))
    lich_dots = s.mr(s.voronoi(P, 38.0, 'F1').outputs['Distance'], 0.5, 0.2)
    lich = s.mul(lich, s.mixf(0.45, 1.0, lich_dots))
    col = s.mixc(s.mul(lich, 0.9), col, lich_col)

    # moss on flat, higher parts
    top_m = s.mr(nz, 0.76, 0.93)
    mpat = s.mr(s.noise(wp, 1.0, 5, 0.6).outputs['Fac'], 0.36, 0.50)
    moss = s.mul(s.mul(top_m, mpat), s.mr(wz, 2.4, 4.0))
    moss_col = s.mixc(s.noise(P, 14.0, 3, 0.5).outputs['Fac'], (0.045, 0.095, 0.014, 1.0), (0.11, 0.165, 0.032, 1.0))
    col = s.mixc(s.mul(moss, 0.95), col, moss_col)

    # splash zone, algae and the wet band near the waterline (z < 1.5)
    zj = s.add(wz, s.mul(s.sub(s.noise(P, 1.4, 4, 0.55).outputs['Fac'], 0.5), 1.5))
    salt = s.mul(s.mul(s.mr(zj, 1.6, 2.4), s.inv(s.mr(zj, 3.3, 4.8))),
                 s.mr(s.noise(P, 2.5, 4, 0.55).outputs['Fac'], 0.35, 0.65))
    col = s.mixc(s.mul(salt, 0.45), col, (0.20, 0.195, 0.185, 1.0))
    algae = s.mul(s.inv(s.mr(zj, -0.8, 1.1)), s.mr(s.noise(P, 5.0, 4, 0.6).outputs['Fac'], 0.35, 0.6))
    col = s.mixc(s.mul(algae, 0.7), col, (0.020, 0.040, 0.014, 1.0))
    wet = s.inv(s.mr(zj, 0.7, 1.9, 0.0, 1.0, 'SMOOTHSTEP'))
    wet_dark = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(wet_dark, 0, s.mul(wet, 0.8))
    s.set(wet_dark, 6, col)
    s.set(wet_dark, 7, (0.38, 0.38, 0.40, 1.0))
    col = wet_dark.outputs[2]

    # roughness
    rough = s.mr(s.noise(P, 6.0, 4, 0.6).outputs['Fac'], 0.3, 0.7, 0.62, 0.92)
    rough = s.mixf(s.mul(lich, 0.8), rough, 0.92)
    rough = s.mixf(s.mul(moss, 0.9), rough, 0.95)
    rough = s.mixf(wet, rough, 0.16)

    nrm = s.bump(height, 1.0, 0.35)
    bsdf = s.principled(Base_Color=col, Roughness=rough, Normal=nrm, Coat_Weight=s.mul(wet, 0.55),
                        Coat_Roughness=0.06, Specular_IOR_Level=0.5)
    s.displacement(height, 0.42, 0.30)
    m = s.finish(bsdf)
    for attr in ('displacement_method',):
        try:
            setattr(m, attr, 'BOTH')
        except Exception:
            pass
    return m


def mat_tower_paint():
    """White painted weathered masonry with two red bands (object coords, origin = tower base)."""
    s = Sh(PFX + "TowerPaint")
    tc = s.tex_coord()
    P = tc.outputs['Object']
    px, py, z = s.sepxyz(P)
    geo = s.geometry()
    nz = s.sepxyz(geo.outputs['Normal'])[2]

    ang = s.math('ARCTAN2', py, px)
    n_blocks = 18.0
    u = s.mul(ang, n_blocks / (2.0 * math.pi))
    bv = s.comb(u, z, 0.0)
    mask_b = s.brick(bv, 1.0, 1.0, 0.5, 0.022, 0.06, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 1))
    brick_m = s.gray(mask_b.outputs['Color'])             # 1 = block face, 0 = mortar
    stone_b = s.brick(bv, 1.0, 1.0, 0.5, 0.022, 0.06, (0.17, 0.16, 0.15, 1), (0.30, 0.28, 0.25, 1),
                      (0.42, 0.40, 0.36, 1), bias=0.0)
    stone = s.mixc(0.25, stone_b.outputs['Color'], (0.12, 0.11, 0.10, 1))

    # band masks with wobbly edges
    zj = s.add(z, s.mul(s.sub(s.noise(P, 2.5, 3, 0.5, 0.3).outputs['Fac'], 0.5), 0.10))

    def band(z0, z1):
        up = s.mr(zj, z0, z0 + 0.04)
        dn = s.mr(zj, z1, z1 + 0.04)
        return s.mn(up, s.inv(dn))
    red = s.mx(band(5.0, 8.6), band(11.4, 15.0))

    white = s.mixc(s.noise(P, 1.3, 3, 0.5).outputs['Fac'], (0.66, 0.65, 0.61, 1), (0.78, 0.77, 0.73, 1))
    redc = s.mixc(s.noise(P, 1.7, 3, 0.5).outputs['Fac'], (0.36, 0.030, 0.022, 1), (0.52, 0.050, 0.034, 1))
    paint = s.mixc(red, white, redc)

    # mortar lines get dirty
    paint = s.mixc(s.mul(s.inv(brick_m), 0.30), paint, s.mixc(red, (0.30, 0.27, 0.24, 1), (0.16, 0.045, 0.03, 1)))
    # vertical dirt streaks (seamless: 3D noise stretched along z)
    st_vec = s.mapping(P, (1.7, 1.7, 0.16))
    streak = s.ramp(s.noise(st_vec, 1.8, 5, 0.6, 0.25).outputs['Fac'],
                    [(0.40, (0, 0, 0, 1)), (0.62, (1, 1, 1, 1))])
    st2 = s.ramp(s.noise(s.mapping(P, (6.0, 6.0, 0.7)), 2.0, 4, 0.55).outputs['Fac'],
                 [(0.45, (0, 0, 0, 1)), (0.7, (1, 1, 1, 1))])
    low = s.mr(z, 16.0, 1.0, 0.15, 1.0)
    dirt = s.mul(s.mixf(0.5, streak, st2), low)
    paint = s.mixc(s.mul(dirt, 0.55), paint, (0.085, 0.07, 0.056, 1))
    # rust drips below the gallery and from the corbels
    drip_zone = s.mr(z, 12.5, 17.4)
    drip = s.ramp(s.noise(s.mapping(P, (9.0, 9.0, 0.25)), 1.5, 3, 0.5).outputs['Fac'],
                  [(0.5, (0, 0, 0, 1)), (0.7, (1, 1, 1, 1))])
    paint = s.mixc(s.mul(s.mul(drip, drip_zone), 0.7), paint, (0.15, 0.065, 0.03, 1))
    # algae / salt crust at the base
    alg = s.mul(s.mr(z, 3.8, 1.0), s.mr(s.noise(P, 2.0, 5, 0.6).outputs['Fac'], 0.35, 0.6))
    paint = s.mixc(s.mul(alg, 0.75), paint, (0.04, 0.055, 0.03, 1))

    # chipped paint reveals stone
    chip_low = s.mr(s.noise(P, 1.2, 3, 0.5).outputs['Fac'], 0.50, 0.70)
    chip_hi = s.mr(s.noise(P, 16.0, 8, 0.65).outputs['Fac'], 0.58, 0.62)
    chip_more = s.mr(z, 9.0, 1.0, 0.15, 1.0)
    edge_chip = s.mul(s.inv(brick_m), s.mr(s.noise(P, 30.0, 3, 0.5).outputs['Fac'], 0.58, 0.68))
    chip = s.mx(s.mul(s.mul(chip_low, chip_hi), chip_more), s.mul(edge_chip, 0.25))
    col = s.mixc(chip, paint, stone)

    # raw stone: plinth below the shaft and the upward-facing gallery slab
    raw = s.mx(s.inv(s.mr(z, Z_SH0 - 0.05, Z_SH0 + 0.02)), s.mul(s.mr(nz, 0.6, 0.9), s.mr(z, 18.3, 18.5)))
    col = s.mixc(raw, col, s.mixc(0.3, stone, (0.13, 0.125, 0.115, 1)))

    # damp, dark base
    damp = s.mr(z, 3.0, 0.9, 0.0, 0.55)
    wetc = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(wetc, 0, damp)
    s.set(wetc, 6, col)
    s.set(wetc, 7, (0.5, 0.5, 0.52, 1))
    col = wetc.outputs[2]

    rough = s.mixf(chip, s.mixf(s.mul(dirt, 0.8), 0.50, 0.82), 0.92)
    rough = s.mixf(raw, rough, 0.88)
    rough = s.mixf(damp, rough, 0.35)

    hgt = s.add(s.mul(brick_m, 0.55), s.mul(s.noise(P, 40.0, 5, 0.6).outputs['Fac'], 0.30))
    hgt = s.sub(hgt, s.mul(chip, 0.35))
    nrm = s.bump(hgt, 0.6, 0.04)
    bsdf = s.principled(Base_Color=col, Roughness=rough, Normal=nrm, Specular_IOR_Level=0.4)
    return s.finish(bsdf)


def mat_granite():
    """Pale dressed granite for sills, lintels, steps, chimney caps."""
    s = Sh(PFX + "Granite")
    tc = s.tex_coord()
    P = tc.outputs['Object']
    px, py, z = s.sepxyz(P)
    geo = s.geometry()
    nz = s.sepxyz(geo.outputs['Normal'])[2]
    base = s.ramp(s.noise(P, 3.0, 5, 0.55).outputs['Fac'], [(0.3, (0.30, 0.285, 0.265, 1)), (0.7, (0.46, 0.44, 0.41, 1))])
    fl = s.mul(s.mr(s.voronoi(P, 90.0, 'F1').outputs['Distance'], 0.4, 0.1), 0.5)
    col = s.mixc(fl, base, (0.12, 0.11, 0.11, 1))
    crack = s.mr(s.voronoi(P, 2.5, 'DISTANCE_TO_EDGE').outputs['Distance'], 0.0, 0.03, 1.0, 0.0)
    col = s.mixc(s.mul(crack, 0.6), col, (0.02, 0.02, 0.02, 1))
    dirt = s.mul(s.mr(s.noise(s.mapping(P, (3, 3, 0.4)), 2.0, 4, 0.55).outputs['Fac'], 0.45, 0.7), 0.5)
    col = s.mixc(dirt, col, (0.07, 0.06, 0.05, 1))
    lich = s.mul(s.mr(nz, 0.6, 0.9), s.mr(s.noise(P, 6.0, 4, 0.6).outputs['Fac'], 0.52, 0.64))
    col = s.mixc(s.mul(lich, 0.6), col, (0.30, 0.32, 0.20, 1))
    hgt = s.add(s.mul(s.noise(P, 60.0, 5, 0.6).outputs['Fac'], 0.6), s.mul(crack, -1.0))
    nrm = s.bump(hgt, 0.6, 0.02)
    bsdf = s.principled(Base_Color=col, Roughness=s.mr(s.noise(P, 8.0, 3, 0.5).outputs['Fac'], 0.3, 0.7, 0.6, 0.9),
                        Normal=nrm)
    return s.finish(bsdf)


def mat_metal():
    """Black painted iron with rust blooms."""
    s = Sh(PFX + "BlackIron")
    P = s.tex_coord().outputs['Object']
    n = s.noise(P, 5.0, 5, 0.6).outputs['Fac']
    rust = s.mul(s.mr(n, 0.60, 0.74), s.mr(s.noise(P, 18.0, 5, 0.6).outputs['Fac'], 0.50, 0.66))
    col = s.ramp(s.noise(P, 30.0, 3, 0.5).outputs['Fac'], [(0.0, (0.014, 0.014, 0.016, 1)), (1.0, (0.034, 0.034, 0.038, 1))])
    col = s.mixc(rust, col, (0.20, 0.075, 0.032, 1))
    rough = s.mixf(rust, s.mr(s.noise(P, 12.0, 4, 0.6).outputs['Fac'], 0.3, 0.7, 0.30, 0.55), 0.85)
    hgt = s.add(s.mul(s.noise(P, 90.0, 4, 0.6).outputs['Fac'], 0.5), rust)
    nrm = s.bump(hgt, 0.35, 0.01)
    bsdf = s.principled(Base_Color=col, Roughness=rough, Metallic=s.mixf(rust, 0.85, 0.0), Normal=nrm,
                        Specular_IOR_Level=0.6)
    return s.finish(bsdf)


def mat_steel():
    s = Sh(PFX + "Steel")
    P = s.tex_coord().outputs['Object']
    n = s.noise(P, 40.0, 4, 0.6).outputs['Fac']
    bsdf = s.principled(Base_Color=(0.55, 0.56, 0.57, 1), Metallic=1.0, Roughness=s.mr(n, 0.3, 0.7, 0.28, 0.5))
    return s.finish(bsdf)


def mat_brass():
    s = Sh(PFX + "Brass")
    P = s.tex_coord().outputs['Object']
    n = s.noise(P, 25.0, 4, 0.6).outputs['Fac']
    col = s.mixc(s.mr(n, 0.4, 0.7), (0.62, 0.40, 0.12, 1), (0.40, 0.26, 0.09, 1))
    bsdf = s.principled(Base_Color=col, Metallic=1.0, Roughness=s.mr(n, 0.3, 0.7, 0.22, 0.4))
    return s.finish(bsdf)


def mat_glass_lantern():
    """Single-surface glass: transparent / fresnel glossy mix (passes shadow rays, no refraction)."""
    s = Sh(PFX + "LanternGlass")
    fr = s.new('ShaderNodeFresnel')
    s.set(fr, 'IOR', 1.5)
    gl = s.new('ShaderNodeBsdfGlossy')
    s.set(gl, 'Roughness', 0.02)
    s.set(gl, 'Color', (0.95, 0.97, 1.0, 1))
    tr = s.new('ShaderNodeBsdfTransparent')
    s.set(tr, 'Color', (0.92, 0.95, 0.93, 1))
    mix = s.new('ShaderNodeMixShader')
    s.set(mix, 0, s.mx(fr.outputs[0], 0.04))
    s.nt.links.new(tr.outputs[0], mix.inputs[1])
    s.nt.links.new(gl.outputs[0], mix.inputs[2])
    # faint warm glow so the lantern reads as lit from inside
    em = s.new('ShaderNodeEmission')
    bb = s.new('ShaderNodeBlackbody')
    s.set(bb, 'Temperature', 2700.0)
    s.nt.links.new(bb.outputs[0], em.inputs['Color'])
    s.set(em, 'Strength', 0.6)
    add = s.new('ShaderNodeAddShader')
    s.nt.links.new(mix.outputs[0], add.inputs[0])
    s.nt.links.new(em.outputs[0], add.inputs[1])
    return s.finish(add.outputs[0])


def mat_lens():
    """Fresnel lens drum: glass, glowing from the lamp inside."""
    s = Sh(PFX + "LampLens")
    fr = s.new('ShaderNodeFresnel')
    s.set(fr, 'IOR', 1.5)
    gl = s.new('ShaderNodeBsdfGlossy')
    s.set(gl, 'Roughness', 0.05)
    tr = s.new('ShaderNodeBsdfTransparent')
    mix = s.new('ShaderNodeMixShader')
    s.set(mix, 0, s.mx(fr.outputs[0], 0.06))
    s.nt.links.new(tr.outputs[0], mix.inputs[1])
    s.nt.links.new(gl.outputs[0], mix.inputs[2])
    em = s.new('ShaderNodeEmission')
    bb = s.new('ShaderNodeBlackbody')
    s.set(bb, 'Temperature', 2700.0)
    s.nt.links.new(bb.outputs[0], em.inputs['Color'])
    geo = s.geometry()
    facing = s.math('ABSOLUTE', s.dot(geo.outputs['Normal'], geo.outputs['Incoming']))
    s.set(em, 'Strength', s.mixf(facing, 12.0, 3.0))
    add = s.new('ShaderNodeAddShader')
    s.nt.links.new(mix.outputs[0], add.inputs[0])
    s.nt.links.new(em.outputs[0], add.inputs[1])
    return s.finish(add.outputs[0])


def mat_lamp_core():
    s = Sh(PFX + "LampCore")
    bb = s.new('ShaderNodeBlackbody')
    s.set(bb, 'Temperature', 2700.0)
    em = s.new('ShaderNodeEmission')
    s.nt.links.new(bb.outputs[0], em.inputs['Color'])
    s.set(em, 'Strength', 120.0)
    return s.finish(em.outputs[0])


def mat_window_dark():
    s = Sh(PFX + "WindowDark")
    P = s.tex_coord().outputs['Object']
    n = s.noise(P, 6.0, 3, 0.5).outputs['Fac']
    bsdf = s.principled(Base_Color=(0.006, 0.008, 0.010, 1), Roughness=s.mr(n, 0.3, 0.7, 0.03, 0.12),
                        Specular_IOR_Level=0.9)
    return s.finish(bsdf)


def mat_window_lit():
    s = Sh(PFX + "WindowLit")
    P = s.tex_coord().outputs['Object']
    n = s.noise(P, 3.0, 2, 0.5).outputs['Fac']
    bb = s.new('ShaderNodeBlackbody')
    s.set(bb, 'Temperature', 2500.0)
    bsdf = s.principled(Base_Color=(0.01, 0.01, 0.012, 1), Roughness=0.05,
                        Emission_Color=bb.outputs[0], Emission_Strength=s.mr(n, 0.3, 0.7, 1.4, 3.0))
    return s.finish(bsdf)


def mat_wood_white():
    s = Sh(PFX + "WoodWhite")
    P = s.tex_coord().outputs['Object']
    px, py, z = s.sepxyz(P)
    n = s.noise(s.mapping(P, (2.0, 2.0, 25.0)), 8.0, 4, 0.55).outputs['Fac']
    col = s.mixc(s.mr(n, 0.35, 0.65), (0.50, 0.49, 0.45, 1), (0.72, 0.71, 0.67, 1))
    flake = s.mr(s.noise(P, 22.0, 6, 0.65).outputs['Fac'], 0.58, 0.62)
    col = s.mixc(flake, col, (0.09, 0.06, 0.04, 1))
    dirt = s.mul(s.mr(s.noise(s.mapping(P, (4, 4, 0.5)), 2.0, 4, 0.5).outputs['Fac'], 0.4, 0.7), 0.5)
    col = s.mixc(dirt, col, (0.08, 0.07, 0.06, 1))
    nrm = s.bump(s.add(n, s.mul(flake, -1.0)), 0.5, 0.01)
    bsdf = s.principled(Base_Color=col, Roughness=s.mixf(flake, 0.55, 0.85), Normal=nrm)
    return s.finish(bsdf)


def mat_clay():
    s = Sh(PFX + "Clay")
    P = s.tex_coord().outputs['Object']
    n = s.noise(P, 20.0, 5, 0.6).outputs['Fac']
    col = s.mixc(s.mr(n, 0.35, 0.65), (0.20, 0.06, 0.03, 1), (0.34, 0.11, 0.05, 1))
    bsdf = s.principled(Base_Color=col, Roughness=0.8, Normal=s.bump(n, 0.5, 0.01))
    return s.finish(bsdf)


def mat_door(name, color):
    """Planked painted door.  Local coords: y = width, z = height."""
    s = Sh(PFX + name)
    P = s.tex_coord().outputs['Object']
    px, py, z = s.sepxyz(P)
    vec = s.comb(py, z, 0.0)
    planks = s.brick(vec, 1.0, 0.17, 4.0, 0.008, 0.01, (1, 1, 1, 1), (0.8, 0.8, 0.8, 1), (0, 0, 0, 1),
                     offset=0.0)
    pm = s.gray(planks.outputs['Color'])
    grain = s.noise(s.mapping(P, (30.0, 4.0, 2.0)), 3.0, 6, 0.6).outputs['Fac']
    tint = s.mr(grain, 0.3, 0.7, 0.6, 1.2)
    base = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(base, 0, 1.0)
    s.set(base, 6, color)
    s.set(base, 7, s.comb(tint, tint, tint))
    col = base.outputs[2]
    col = s.mixc(s.mul(s.inv(pm), 0.9), col, (0.01, 0.008, 0.008, 1))
    wear = s.mul(s.mr(s.noise(P, 9.0, 6, 0.65).outputs['Fac'], 0.55, 0.64), s.mr(z, 1.2, 0.0, 0.3, 1.0))
    col = s.mixc(wear, col, (0.20, 0.15, 0.11, 1))
    nrm = s.bump(s.add(pm, s.mul(grain, 0.4)), 0.8, 0.01)
    bsdf = s.principled(Base_Color=col, Roughness=s.mixf(wear, 0.5, 0.85), Normal=nrm)
    return s.finish(bsdf)


def mat_slate():
    s = Sh(PFX + "Slate")
    P = s.tex_coord().outputs['Object']
    px, py, z = s.sepxyz(P)
    v = s.mul(s.math('ABSOLUTE', py), 1.29)
    vec = s.comb(px, v, 0.0)
    row_h, w = 0.22, 0.26
    b = s.brick(vec, 1.0, w, row_h, 0.012, 0.01, (0.040, 0.044, 0.052, 1), (0.085, 0.09, 0.10, 1),
                (0.01, 0.01, 0.01, 1), offset=0.5)
    mask = s.gray(s.brick(vec, 1.0, w, row_h, 0.012, 0.01, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 1)).outputs['Color'])
    rowf = s.math('FRACT', s.math('DIVIDE', v, row_h))
    shade = s.mr(rowf, 0.0, 0.35, 0.55, 1.0)               # darker at the lower (overlapped) edge
    col = b.outputs['Color']
    sh_n = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(sh_n, 0, 1.0)
    s.set(sh_n, 6, col)
    s.set(sh_n, 7, s.comb(shade, shade, shade))
    col = sh_n.outputs[2]
    wea = s.mr(s.noise(P, 2.0, 4, 0.55).outputs['Fac'], 0.4, 0.7)
    col = s.mixc(s.mul(wea, 0.35), col, (0.12, 0.12, 0.11, 1))
    lich = s.mul(s.mr(s.noise(P, 7.0, 5, 0.6).outputs['Fac'], 0.55, 0.68), s.mr(s.noise(P, 1.5, 2, 0.5).outputs['Fac'], 0.45, 0.6))
    col = s.mixc(s.mul(lich, 0.7), col, (0.20, 0.22, 0.10, 1))
    hgt = s.add(s.mul(mask, 0.5), s.mul(rowf, 0.5))
    nrm = s.bump(s.add(hgt, s.mul(s.noise(P, 60.0, 5, 0.6).outputs['Fac'], 0.25)), 1.0, 0.04)
    bsdf = s.principled(Base_Color=col, Roughness=s.mr(wea, 0.3, 0.7, 0.45, 0.75), Normal=nrm,
                        Specular_IOR_Level=0.6)
    return s.finish(bsdf)


def mat_keeper_stone():
    """Rubble masonry (voronoi cells + lime mortar) for the keeper's house."""
    s = Sh(PFX + "KeeperStone")
    P = s.tex_coord().outputs['Object']
    px, py, z = s.sepxyz(P)
    u = s.add(px, py)
    vec = s.comb(u, s.mul(z, 1.5), 0.0)
    edge = s.voronoi(vec, 2.4, 'DISTANCE_TO_EDGE').outputs['Distance']
    cells = s.voronoi(vec, 2.4, 'F1')
    cv = s.gray(cells.outputs['Color'])
    joint = s.mr(edge, 0.03, 0.10, 1.0, 0.0)                # 1 in the joint
    puff = s.mr(edge, 0.0, 0.28, 0.0, 1.0, 'SMOOTHSTEP')
    stone = s.ramp(cv, [(0.0, (0.15, 0.145, 0.135, 1)), (0.5, (0.27, 0.255, 0.23, 1)), (1.0, (0.40, 0.37, 0.33, 1))])
    var = s.mr(s.noise(P, 6.0, 5, 0.6).outputs['Fac'], 0.3, 0.7, 0.75, 1.15)
    sv = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(sv, 0, 1.0)
    s.set(sv, 6, stone)
    s.set(sv, 7, s.comb(var, var, var))
    col = sv.outputs[2]
    col = s.mixc(joint, col, (0.36, 0.34, 0.30, 1))
    dirt = s.mul(s.mr(s.noise(s.mapping(P, (2, 2, 0.35)), 1.8, 5, 0.6).outputs['Fac'], 0.4, 0.68), 0.6)
    col = s.mixc(dirt, col, (0.05, 0.045, 0.04, 1))
    algae = s.mul(s.mr(z, 1.4, -0.2), s.mr(s.noise(P, 3.0, 5, 0.6).outputs['Fac'], 0.35, 0.6))
    col = s.mixc(s.mul(algae, 0.8), col, (0.03, 0.05, 0.02, 1))
    lich = s.mul(s.mr(s.noise(P, 5.0, 5, 0.6).outputs['Fac'], 0.58, 0.68), 0.7)
    col = s.mixc(lich, col, (0.45, 0.40, 0.14, 1))
    damp = s.mr(z, 1.2, 0.0, 0.0, 0.5)
    wetc = s.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    s.set(wetc, 0, damp)
    s.set(wetc, 6, col)
    s.set(wetc, 7, (0.5, 0.5, 0.52, 1))
    col = wetc.outputs[2]
    hgt = s.add(s.mul(puff, 0.8), s.mul(s.noise(P, 25.0, 6, 0.65).outputs['Fac'], 0.35))
    nrm = s.bump(hgt, 1.2, 0.06)
    bsdf = s.principled(Base_Color=col, Roughness=s.mixf(joint, 0.88, 0.95), Normal=nrm)
    return s.finish(bsdf)


def mat_beam():
    s = Sh(PFX + "BeamGlow")
    tc = s.tex_coord()
    x = s.sepxyz(tc.outputs['Object'])[0]
    geo = s.geometry()
    t = s.mr(x, 0.0, BEAM_LEN, 0.0, 1.0)
    fade_len = s.pw(s.inv(t), 1.7)
    fade_in = s.mr(x, 1.0, 14.0, 0.0, 1.0, 'SMOOTHSTEP')
    facing = s.math('ABSOLUTE', s.dot(geo.outputs['Normal'], geo.outputs['Incoming']))
    edge = s.pw(facing, 1.4)
    vt = s.new('ShaderNodeVectorTransform', vector_type='VECTOR', convert_from='OBJECT', convert_to='WORLD')
    s.set(vt, 'Vector', (1.0, 0.0, 0.0))
    align = s.math('ABSOLUTE', s.dot(vt.outputs[0], geo.outputs['Incoming']))
    dirf = s.add(0.35, s.mul(s.pw(align, 3.0), 1.65))      # brighter when pointing at the viewer
    strength = s.mul(s.mul(s.mul(fade_len, fade_in), s.mul(edge, dirf)), 1.2)
    em = s.new('ShaderNodeEmission')
    s.set(em, 'Color', (1.0, 0.80, 0.52, 1.0))
    s.set(em, 'Strength', strength)
    tr = s.new('ShaderNodeBsdfTransparent')
    add = s.new('ShaderNodeAddShader')
    s.nt.links.new(em.outputs[0], add.inputs[0])
    s.nt.links.new(tr.outputs[0], add.inputs[1])
    return s.finish(add.outputs[0])


# --------------------------------------------------------------------------------------
# Mesh builder
# --------------------------------------------------------------------------------------
class MB:
    """bmesh wrapper: every primitive is tagged with the current material index."""

    def __init__(self):
        self.bm = bmesh.new()
        self.mi = 0

    def _tag(self, n0, smooth=None):
        self.bm.faces.ensure_lookup_table()
        for f in self.bm.faces[n0:]:
            f.material_index = self.mi
            if smooth is not None:
                f.smooth = smooth

    def box(self, M, sx, sy, sz):
        n0 = len(self.bm.faces)
        bmesh.ops.create_cube(self.bm, size=1.0, matrix=M @ Matrix.Diagonal((sx, sy, sz, 1.0)))
        self._tag(n0, False)

    def cyl(self, M, r1, r2, h, seg=8, cap=True, smooth=True):
        n0 = len(self.bm.faces)
        bmesh.ops.create_cone(self.bm, cap_ends=cap, cap_tris=False, segments=seg, radius1=r1, radius2=r2,
                              depth=h, matrix=M)
        self._tag(n0, smooth)

    def rod(self, p0, p1, r, seg=6):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        L = d.length
        if L < 1e-6:
            return
        rot = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
        self.cyl(Matrix.Translation((p0 + p1) * 0.5) @ rot, r, r, L, seg, True, True)

    def sphere(self, M, r, us=24, vs=12):
        n0 = len(self.bm.faces)
        bmesh.ops.create_uvsphere(self.bm, u_segments=us, v_segments=vs, radius=r, matrix=M)
        self._tag(n0, True)

    def lathe(self, prof, seg=48, M=None, close=False, smooth=True):
        """prof: [(r, z, sharp)] ordered so that the outside is on the right of travel."""
        bm = self.bm
        n0 = len(bm.faces)
        rings, allv = [], []
        for p in prof:
            r, z = p[0], p[1]
            if abs(r) < 1e-6:
                v = bm.verts.new((0.0, 0.0, z))
                rings.append([v] * seg)
                allv.append(v)
            else:
                ring = [bm.verts.new((r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg), z))
                        for j in range(seg)]
                rings.append(ring)
                allv.extend(ring)
        npairs = len(rings) if close else len(rings) - 1
        for i in range(npairs):
            a, b = rings[i], rings[(i + 1) % len(rings)]
            for j in range(seg):
                j2 = (j + 1) % seg
                vs = []
                for v in (a[j], a[j2], b[j2], b[j]):
                    if not vs or vs[-1] != v:
                        vs.append(v)
                if len(vs) > 1 and vs[0] == vs[-1]:
                    vs.pop()
                if len(vs) < 3:
                    continue
                try:
                    bm.faces.new(vs)
                except ValueError:
                    pass
        for i, p in enumerate(prof):
            if len(p) > 2 and p[2] and abs(p[0]) > 1e-6:
                ring = rings[i]
                for j in range(seg):
                    e = bm.edges.get((ring[j], ring[(j + 1) % seg]))
                    if e is not None:
                        e.smooth = False
        if M is not None:
            bmesh.ops.transform(bm, matrix=M, verts=allv)
        self._tag(n0, smooth)

    def prism(self, prof_yz, x0, x1, M=None):
        """Closed prism: polygon in the (y, z) plane extruded along x."""
        bm = self.bm
        n0 = len(bm.faces)
        va = [bm.verts.new((x0, y, z)) for (y, z) in prof_yz]
        vb = [bm.verts.new((x1, y, z)) for (y, z) in prof_yz]
        n = len(prof_yz)
        faces = [bm.faces.new(va[::-1]), bm.faces.new(vb)]
        for i in range(n):
            j = (i + 1) % n
            faces.append(bm.faces.new((va[i], va[j], vb[j], vb[i])))
        bmesh.ops.recalc_face_normals(bm, faces=faces)
        if M is not None:
            bmesh.ops.transform(bm, matrix=M, verts=va + vb)
        self._tag(n0, False)

    def to_object(self, name, coll, mats, loc=(0, 0, 0), rot=(0, 0, 0)):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        for m in mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        return ob


def add_bevel(ob, width=0.02, segments=2, angle=35.0):
    m = ob.modifiers.new("Bevel", 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = 'ANGLE'
    m.angle_limit = math.radians(angle)


def add_cut(ob, cutter):
    m = ob.modifiers.new("Cut", 'BOOLEAN')
    m.operation = 'DIFFERENCE'
    m.operand_type = 'OBJECT'
    m.object = cutter
    m.solver = 'EXACT'


def make_cutter(name, coll, mb, loc=(0, 0, 0), rot=(0, 0, 0)):
    ob = mb.to_object(name, coll, [], loc, rot)
    ob.display_type = 'WIRE'
    ob.hide_render = True
    return ob


# --------------------------------------------------------------------------------------
# Islet
# --------------------------------------------------------------------------------------
def nfbm(x, y, z, octaves, freq, gain=0.5):
    s, a, f, tot = 0.0, 1.0, freq, 0.0
    for _ in range(octaves):
        s += a * mnoise.noise(Vector((x * f, y * f, z * f)))
        tot += a
        a *= gain
        f *= 2.03
    return s / tot


def pad_weight(dx, dy):
    d_t = math.hypot(dx - TOWER_REL.x, dy - TOWER_REL.y) - 3.9
    c, s_ = math.cos(HOUSE_ROT), math.sin(HOUSE_ROT)
    lx = (dx - HOUSE_REL.x) * c + (dy - HOUSE_REL.y) * s_
    ly = -(dx - HOUSE_REL.x) * s_ + (dy - HOUSE_REL.y) * c
    qx = abs(lx) - HOUSE_W / 2 - 0.4
    qy = abs(ly) - HOUSE_D / 2 - 0.4
    d_h = math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0)
    return 1.0 - smoothstep(0.0, 3.4, min(d_t, d_h))


def islet_height(dx, dy):
    n_low = nfbm(dx, dy, 3.7, 2, 0.075)
    n_mid = nfbm(dx, dy, 9.1, 2, 0.22)
    t = math.hypot(dx / ISLET_RX, dy / ISLET_RY) * (1.0 + 0.32 * n_low + 0.07 * n_mid)
    if t < 0.55:
        p = PLATEAU
    elif t <= 1.0:
        p = PLATEAU * (1.0 - ((t - 0.55) / 0.45) ** 1.8)
    else:
        p = -24.0 * (t - 1.0)
    p = max(p, BOTTOM)
    wd = smoothstep(0.42, 0.85, t)
    d1 = nfbm(dx, dy, p * 0.35, 4, 0.20)
    rg = 1.0 - abs(nfbm(dx, dy, p * 0.5 + 20.0, 3, 0.33)) * 2.2
    d2 = nfbm(dx, dy, p * 1.1 + 5.0, 3, 1.1)
    plat = nfbm(dx, dy, 2.2, 3, 0.45)
    p += wd * (1.5 * d1 + 0.45 * rg + 0.30 * d2) + (1.0 - wd) * 0.35 * plat
    # ledge / strata look on the cliffs
    q = p / 0.9
    p = p + wd * 0.18 * (math.sin(q * 2.0 * math.pi) * 0.5)
    w = pad_weight(dx, dy)
    pad = PAD + 0.10 * nfbm(dx, dy, 5.5, 3, 0.6)
    p = p * (1.0 - w) + pad * w
    return max(p, BOTTOM)


def build_islet(coll, mat_rock_):
    nr, ns = 170, 360
    rho_max = 1.75
    bm = bmesh.new()
    rings = []
    for i in range(nr + 1):
        rho = rho_max * (i / nr)
        if i == 0:
            v = bm.verts.new((0.0, 0.0, islet_height(0.0, 0.0)))
            rings.append([v] * ns)
            continue
        ring = []
        for j in range(ns):
            a = 2 * math.pi * j / ns
            dx = ISLET_RX * rho * math.cos(a)
            dy = ISLET_RY * rho * math.sin(a)
            z = islet_height(dx, dy) if i < nr else BOTTOM
            ring.append(bm.verts.new((dx, dy, z)))
        rings.append(ring)
    for i in range(nr):
        a, b = rings[i], rings[i + 1]
        for j in range(ns):
            j2 = (j + 1) % ns
            if i == 0:
                bm.faces.new((a[0], b[j2], b[j]))      # outward normal = up
            else:
                bm.faces.new((a[j], a[j2], b[j2], b[j]))
    # careful with orientation: rings grow outward (+r), angle grows CCW -> (a[j], b[j], b[j2], a[j2]) is up
    bm.faces.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # make sure the surface normals point up/out
    top = max(bm.faces, key=lambda f: f.calc_center_median().z)
    if top.normal.z < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    # bottom cap
    last = rings[nr]
    cap = bm.faces.new(last[::-1])
    bm.normal_update()
    if cap.normal.z > 0:
        cap.normal_flip()
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(PFX + "Islet")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat_rock_)
    ob = bpy.data.objects.new(PFX + "Islet", me)
    coll.objects.link(ob)
    ob.location = ISLET_C
    return ob


def build_boulders(coll, mat_rock_):
    rnd = random.Random(7)
    bm = bmesh.new()
    spots = []
    for k in range(22):
        a = rnd.uniform(0, 2 * math.pi)
        t = rnd.uniform(0.82, 1.14) if k < 16 else rnd.uniform(0.55, 0.75)
        dx = ISLET_RX * t * math.cos(a) * 1.0
        dy = ISLET_RY * t * math.sin(a) * 1.0
        spots.append((dx, dy))
    for (dx, dy) in spots:
        if pad_weight(dx, dy) > 0.3:
            continue
        r = rnd.uniform(0.5, 1.5)
        sx, sy, sz = r * rnd.uniform(0.9, 1.5), r * rnd.uniform(0.9, 1.5), r * rnd.uniform(0.55, 0.95)
        gz = islet_height(dx, dy)
        M = Matrix.Translation((dx, dy, gz - 0.12 * sz)) @ Rz(rnd.uniform(0, 6.28)) @ Matrix.Diagonal((sx, sy, sz, 1))
        res = bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0, matrix=M)
        seed = rnd.uniform(0, 100)
        for v in res['verts']:
            p = v.co
            n = nfbm(p.x, p.y, p.z + seed, 4, 0.9)
            rg = 1.0 - abs(nfbm(p.x, p.y, p.z + seed + 40, 3, 1.6)) * 2.0
            d = (p - Vector((dx, dy, gz))).normalized()
            p += d * (0.28 * n * r + 0.10 * rg * r)
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(PFX + "Boulders")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat_rock_)
    ob = bpy.data.objects.new(PFX + "Boulders", me)
    coll.objects.link(ob)
    ob.location = ISLET_C
    return ob


# --------------------------------------------------------------------------------------
# Tower
# --------------------------------------------------------------------------------------
def shaft_r(z):
    return R_SH0 + (R_SH1 - R_SH0) * (z - Z_SH0) / (Z_SH1 - Z_SH0)


def tower_profile():
    prof = [(0.0, -3.0, 1), (3.6, -3.0, 1), (3.6, 0.74, 1), (3.5, 0.92, 1), (R_SH0, Z_SH0, 1)]
    nz = 10
    for i in range(1, nz):
        z = Z_SH0 + (Z_SH1 - Z_SH0) * i / nz
        prof.append((shaft_r(z), z, 0))
    prof += [(R_SH1, Z_SH1, 1),
             (2.50, Z_SH1, 1), (2.50, 17.70, 1), (2.85, 17.70, 1), (2.85, 18.00, 1), (3.25, 18.00, 1),
             (3.25, 18.42, 1), (3.17, Z_GAL, 1),
             (1.75, Z_GAL, 1), (1.75, 19.10, 1), (1.66, 19.20, 1), (0.0, 19.20, 1)]
    return prof


def frame_M(az, d, z, tilt=0.0):
    """Frame on the tower wall: x outward (radial), y tangent, z up (tilted with the taper)."""
    return (Matrix.Translation((d * math.cos(az), d * math.sin(az), z)) @ Rz(az)
            @ Matrix.Rotation(-tilt, 4, 'Y'))


def add_window(parts, M, w, h, frame_idx):
    """M: frame at the wall surface (x outwards).  parts: dict of MB (cut, trim, frame, glass)."""
    parts['cut'].box(M @ Tr(-0.25, 0, 0), 1.0, w, h)
    g = parts['glass']
    g.box(M @ Tr(-0.34, 0, 0), 0.02, w + 0.04, h + 0.04)
    f = parts['frame']
    f.mi = frame_idx
    t, d = 0.055, 0.07
    f.box(M @ Tr(-0.30, 0, h / 2 - t / 2 + 0.01), d, w + 0.08, t)
    f.box(M @ Tr(-0.30, 0, -h / 2 + t / 2 - 0.01), d, w + 0.08, t)
    f.box(M @ Tr(-0.30, w / 2 - t / 2 + 0.01, 0), d, t, h)
    f.box(M @ Tr(-0.30, -w / 2 + t / 2 - 0.01, 0), d, t, h)
    f.box(M @ Tr(-0.29, 0, 0), 0.05, 0.035, h)
    f.box(M @ Tr(-0.29, 0, h * 0.12), 0.05, w, 0.035)
    tr = parts['trim']
    tr.mi = 0
    tr.box(M @ Tr(0.0, 0, h / 2 + 0.13), 0.34, w + 0.5, 0.17)
    tr.box(M @ Tr(0.02, 0, -h / 2 - 0.045), 0.40, w + 0.32, 0.13)


def add_door_surround(parts, M, w, h):
    tr = parts['trim']
    tr.mi = 0
    jw = 0.24
    tr.box(M @ Tr(0.0, 0, h / 2 + 0.17), 0.36, w + 2 * jw + 0.12, 0.30)
    tr.box(M @ Tr(0.0, w / 2 + jw / 2, 0.0), 0.26, jw, h)
    tr.box(M @ Tr(0.0, -w / 2 - jw / 2, 0.0), 0.26, jw, h)
    parts['cut'].box(M @ Tr(-0.30, 0, 0), 1.0, w, h)


def make_door_object(name, coll, mat_wood, mat_iron, w, h, loc, rot):
    mb = MB()
    mb.mi = 0
    mb.box(Tr(0, 0, h / 2), 0.08, w - 0.02, h - 0.02)
    mb.mi = 1
    for zc in (0.30, h * 0.5, h - 0.30):
        mb.box(Tr(0.045, 0, zc), 0.016, w - 0.06, 0.11)
    mb.box(Tr(0.06, w * 0.32, h * 0.5), 0.05, 0.035, 0.035)
    mb.cyl(Tr(0.075, w * 0.32, h * 0.5) @ Matrix.Rotation(math.pi / 2, 4, 'Y'), 0.035, 0.035, 0.015, 12)
    ob = mb.to_object(name, coll, [mat_wood, mat_iron], loc, rot)
    add_bevel(ob, 0.008, 1)
    return ob


def build_tower(coll, M):
    objs = []
    # --- main body (closed lathe, windows/door cut by boolean) ---
    body = MB()
    body.lathe(tower_profile(), SEG_TOWER)
    tilt = math.atan((R_SH0 - R_SH1) / (Z_SH1 - Z_SH0))

    parts = {k: MB() for k in ('cut', 'trim', 'frame', 'glass')}
    parts['frame'].mi = 0
    # windows spiral up the shaft
    wins = [(5.2, math.radians(-40)), (8.4, math.radians(-160)), (11.9, math.radians(-98)),
            (14.6, math.radians(-25)), (16.3, math.radians(150))]
    for z, az in wins:
        r = shaft_r(z)
        add_window(parts, frame_M(az, r, z, tilt), 0.62, 1.0, 0)
    # door
    door_h, door_w = 2.25, 1.08
    door_z = 0.95 + door_h / 2
    rd = shaft_r(door_z)
    DM = frame_M(DOOR_AZ, rd, door_z, tilt)
    add_door_surround(parts, DM, door_w, door_h)
    # steps toward the door
    tr = parts['trim']
    tr.mi = 0
    for k in range(1, 5):
        top = 0.92 - 0.23 * k
        d_far = 3.5 + 0.34 * k
        d0 = 3.2
        c = (d0 + d_far) / 2
        zc = (top + (-1.2)) / 2
        MS = Rz(DOOR_AZ) @ Tr(c, 0, zc)
        tr.box(MS, d_far - d0, 2.0 + 0.0, top + 1.2)
    # --- lantern & gallery ---
    metal = MB()
    metal.mi = 0
    # gallery railing
    rr = 3.05
    metal.lathe([(rr + 0.03 * math.cos(a), Z_GAL + 1.06 + 0.03 * math.sin(a)) for a in
                 [-i * 2 * math.pi / 8 for i in range(8)]], 64, close=True)
    metal.lathe([(rr + 0.02 * math.cos(a), Z_GAL + 0.58 + 0.02 * math.sin(a)) for a in
                 [-i * 2 * math.pi / 6 for i in range(6)]], 64, close=True)
    metal.lathe([(rr + 0.02 * math.cos(a), Z_GAL + 0.12 + 0.02 * math.sin(a)) for a in
                 [-i * 2 * math.pi / 6 for i in range(6)]], 64, close=True)
    npost = 44
    gap_a = DOOR_AZ + math.pi  # no gap, full ring
    for i in range(npost):
        a = 2 * math.pi * i / npost
        metal.cyl(Tr(rr * math.cos(a), rr * math.sin(a), Z_GAL + 0.54) , 0.018, 0.018, 1.08, 6)
    # corner posts (thicker) every 11
    for i in range(0, npost, 11):
        a = 2 * math.pi * i / npost
        metal.cyl(Tr(rr * math.cos(a), rr * math.sin(a), Z_GAL + 0.62), 0.032, 0.032, 1.24, 8)
    # lantern base sill, mullions, bars, cornice
    LR = 1.62
    nside = 12
    metal.lathe([(0.0, 19.20, 1), (1.72, 19.20, 1), (1.72, 19.36, 1), (1.64, 19.40, 1), (0.0, 19.40, 1)], 48)
    side = 2 * LR * math.sin(math.pi / nside)
    apothem = LR * math.cos(math.pi / nside)
    for k in range(nside):
        a = 2 * math.pi * k / nside
        metal.box(Tr(LR * math.cos(a), LR * math.sin(a), 20.4) @ Rz(a), 0.09, 0.07, 2.04)
    for k in range(nside):
        a = 2 * math.pi * (k + 0.5) / nside
        for zc, hh in ((21.35, 0.08), (20.40, 0.045), (19.46, 0.07)):
            metal.box(Tr(apothem * math.cos(a), apothem * math.sin(a), zc) @ Rz(a), 0.07, side, hh)
    cornice = [(0.0, 21.38, 1), (1.78, 21.38, 1), (2.10, 21.44, 1), (2.14, 21.62, 1), (1.98, 21.74, 1),
               (1.92, 21.80, 1), (0.0, 21.80, 1)]
    metal.lathe(cornice, 48)
    dome = [(1.90, 21.80, 1)]
    for i in range(1, 15):
        a = (math.pi / 2) * i / 14
        dome.append((1.90 * math.cos(a) * (1 - 0.02 * i / 14) + 0.0, 21.80 + 1.35 * math.sin(a), 0))
    dome[-1] = (0.0, 21.80 + 1.35, 0)
    metal.lathe(dome, 48)
    # dome ribs
    for k in range(12):
        a = 2 * math.pi * k / 12
        for i in range(10):
            a0 = (math.pi / 2) * i / 10
            a1 = (math.pi / 2) * (i + 1) / 10
            p0 = Vector((1.93 * math.cos(a0) * math.cos(a), 1.93 * math.cos(a0) * math.sin(a), 21.80 + 1.37 * math.sin(a0)))
            p1 = Vector((1.93 * math.cos(a1) * math.cos(a), 1.93 * math.cos(a1) * math.sin(a), 21.80 + 1.37 * math.sin(a1)))
            if i == 9:
                p1 = Vector((0, 0, 23.18))
            metal.rod(p0, p1, 0.03, 4)
    metal.lathe([(0.0, 23.10, 1), (0.30, 23.10, 1), (0.30, 23.17, 1), (0.10, 23.26, 1), (0.10, 23.45, 0),
                 (0.0, 23.45, 1)], 24)

    # --- lens, pedestal, lamp core, brass, steel (separate material slots) ---
    brass = MB()
    brass.lathe([(0.0, Z_GAL + 0.05, 1), (0.62, Z_GAL + 0.05, 1), (0.62, Z_GAL + 0.20, 1), (0.40, Z_GAL + 0.30, 1),
                 (0.40, 19.42, 1), (0.74, 19.42, 1), (0.74, 19.50, 1), (0.0, 19.50, 1)], 48)
    brass.lathe([(0.0, 21.22, 1), (0.74, 21.22, 1), (0.74, 21.30, 1), (0.40, 21.34, 1), (0.40, 21.7, 1), (0.0, 21.7, 1)], 48)
    brass.sphere(Tr(0, 0, 23.45), 0.20, 24, 12)
    steel = MB()
    steel.lathe([(0.0, 23.55, 1), (0.025, 23.55, 1), (0.012, 25.4, 0), (0.0, 25.5, 1)], 8)
    steel.sphere(Tr(0, 0, 24.0), 0.045, 8, 6)
    lens = MB()
    lp = [(0.0, 19.50, 1)]
    nzig = 22
    z0, z1 = 19.50, 21.22
    for i in range(nzig + 1):
        zz = z0 + (z1 - z0) * i / nzig
        base_r = 0.50 + 0.26 * math.sin(math.pi * (zz - z0) / (z1 - z0))
        lp.append((base_r + (0.045 if i % 2 == 0 else -0.01), zz, 1))
    lp.append((0.0, z1, 1))
    lp[1] = (0.52, z0, 1)
    lens.lathe(lp, 40)
    core = MB()
    core.sphere(Tr(0, 0, Z_LAMP), 0.15, 16, 8)
    glass = MB()
    glass.lathe([(LR, 19.40), (LR, 21.38)], nside, smooth=False)

    # --- assemble tower objects ---
    org = TOWER_ORG
    ob_body = body.to_object(PFX + "Tower", coll, [M['paint']], org)
    cutter = make_cutter(PFX + "TowerCutter", coll, parts['cut'], org)
    add_cut(ob_body, cutter)
    add_bevel(ob_body, 0.02, 2, 40.0)
    ob_trim = parts['trim'].to_object(PFX + "TowerTrim", coll, [M['granite']], org)
    add_bevel(ob_trim, 0.02, 2)
    ob_frames = parts['frame'].to_object(PFX + "TowerWindows", coll, [M['metal']], org)
    ob_wglass = parts['glass'].to_object(PFX + "TowerWindowGlass", coll, [M['wdark']], org)
    # steps railing
    rail = MB()
    rail.mi = 0
    for sgn in (-1, 1):
        y = sgn * 1.08
        pts = []
        for d in (3.55, 4.20, 4.85):
            k = (d - 3.5) / 0.34
            zt = 0.92 - 0.23 * min(max(k, 0), 4)
            pts.append((d, zt))
        for d, zt in pts:
            P0 = Rz(DOOR_AZ) @ Vector((d, y, zt - 0.1))
            P1 = Rz(DOOR_AZ) @ Vector((d, y, zt + 0.92))
            rail.rod(P0, P1, 0.025, 6)
        top0 = Rz(DOOR_AZ) @ Vector((3.5, y, 0.92 + 0.92 + 0.0))
        top1 = Rz(DOOR_AZ) @ Vector((4.9, y, 0.92 - 0.23 * 4.1 + 0.92))
        rail.rod(top0, top1, 0.03, 6)
        mid0 = top0 - Vector((0, 0, 0.45))
        mid1 = top1 - Vector((0, 0, 0.45))
        rail.rod(mid0, mid1, 0.02, 6)
    ob_rail = rail.to_object(PFX + "StairRail", coll, [M['metal']], org)
    ob_metal = metal.to_object(PFX + "LanternFrame", coll, [M['metal']], org)
    ob_brass = brass.to_object(PFX + "LampBrass", coll, [M['brass']], org)
    ob_steel = steel.to_object(PFX + "LightningRod", coll, [M['steel']], org)
    ob_lens = lens.to_object(PFX + "LampLens", coll, [M['lens']], org)
    ob_core = core.to_object(PFX + "LampCore", coll, [M['core']], org)
    ob_core.visible_shadow = False
    ob_lens.visible_shadow = False
    ob_glass = glass.to_object(PFX + "LanternGlass", coll, [M['lglass']], org)
    ob_glass.visible_shadow = False
    # door
    dloc = org + Rz(DOOR_AZ) @ Vector((rd - 0.34, 0, 0.95))
    ob_door = make_door_object(PFX + "TowerDoor", coll, M['door_red'], M['metal'], door_w, door_h - 0.02,
                               dloc, (0, 0, DOOR_AZ))
    for o in (ob_body, ob_trim, ob_frames, ob_wglass, ob_rail, ob_metal, ob_brass, ob_steel, ob_lens, ob_core,
              ob_glass, ob_door):
        objs.append(o)
    return objs


# --------------------------------------------------------------------------------------
# Keeper's house
# --------------------------------------------------------------------------------------
def build_house(coll, M):
    W, D, H = HOUSE_W, HOUSE_D, HOUSE_H
    G = 1.55
    org = HOUSE_ORG
    rot = (0, 0, HOUSE_ROT)
    walls = MB()
    walls.prism([(-D / 2, -1.6), (D / 2, -1.6), (D / 2, H), (0.0, H + G), (-D / 2, H)], -W / 2, W / 2)
    # chimney breast on the +x gable
    cx = W / 2 - 0.30
    walls.box(Tr(cx, 0.0, (-1.6 + H + G + 1.0) / 2), 0.80, 0.80, (H + G + 1.0 + 1.6))

    parts = {k: MB() for k in ('cut', 'trim', 'frame', 'glass')}
    # windows: M frames with x pointing out of the wall
    def wall_M(alpha, x, y, z):
        return Tr(x, y, z) @ Rz(alpha)
    for xs in (-1.85, 1.85):
        add_window(parts, wall_M(-math.pi / 2, xs, -D / 2, 1.55), 0.80, 1.05, 0)
    add_window(parts, wall_M(math.pi / 2, -1.2, D / 2, 1.55), 0.80, 1.05, 0)
    add_window(parts, wall_M(math.pi / 2, 1.2, D / 2, 1.55), 0.80, 1.05, 0)
    add_window(parts, wall_M(math.pi, -W / 2, 0.0, 1.55), 0.70, 1.0, 0)
    add_window(parts, wall_M(math.pi, -W / 2, 0.0, H + 0.55), 0.40, 0.45, 0)
    # door on the front wall
    dh, dw = 2.05, 0.95
    DM = wall_M(-math.pi / 2, 0.0, -D / 2, 0.14 + dh / 2)
    add_door_surround(parts, DM, dw, dh)
    tr = parts['trim']
    tr.mi = 0
    # doorstep
    tr.box(Tr(0.0, -D / 2 - 0.30, -0.15), 1.9, 0.7, 0.62)
    tr.box(Tr(0.0, -D / 2 - 0.85, -0.38), 1.9, 0.5, 0.52)
    # chimney cap & pots
    tr.box(Tr(cx, 0.0, H + G + 1.05), 0.98, 0.98, 0.14)
    tr.box(Tr(cx, 0.0, H + G + 0.92), 0.86, 0.86, 0.12)
    tr.mi = 1
    for yy in (-0.2, 0.2):
        tr.cyl(Tr(cx, yy, H + G + 1.30), 0.14, 0.11, 0.42, 12)
    # base course / plinth stones
    tr.mi = 0
    # slate roof
    roof = MB()
    a = D / 2 + 0.42
    p = G / (D / 2)
    zr = H + G + 0.14
    ze = zr - p * a
    t = 0.16
    roof.prism([(-a, ze), (0.0, zr), (a, ze), (a, ze - t), (0.0, zr - t), (-a, ze - t)], -W / 2 - 0.32, W / 2 + 0.32)
    roof.mi = 0
    roof.box(Tr(0, 0, zr + 0.03), W + 0.70, 0.20, 0.10)

    ob_w = walls.to_object(PFX + "KeeperWalls", coll, [M['kstone']], org, rot)
    cutter = make_cutter(PFX + "KeeperCutter", coll, parts['cut'], org, rot)
    add_cut(ob_w, cutter)
    add_bevel(ob_w, 0.025, 1, 40.0)
    ob_t = parts['trim'].to_object(PFX + "KeeperTrim", coll, [M['granite'], M['clay']], org, rot)
    add_bevel(ob_t, 0.02, 2)
    ob_f = parts['frame'].to_object(PFX + "KeeperWindowFrames", coll, [M['wood']], org, rot)
    # lit panes: two of the windows glow, the rest are dark
    ob_g = parts['glass'].to_object(PFX + "KeeperWindowGlass", coll, [M['wlit']], org, rot)
    ob_r = roof.to_object(PFX + "KeeperRoof", coll, [M['slate']], org, rot)
    add_bevel(ob_r, 0.02, 1, 40.0)
    dpos = org + Rz(HOUSE_ROT) @ Vector((0.0, -D / 2 + 0.30, 0.14))
    ob_d = make_door_object(PFX + "KeeperDoor", coll, M['door_green'], M['metal'], dw, dh - 0.02, dpos,
                            (0, 0, HOUSE_ROT - math.pi / 2))
    return [ob_w, ob_t, ob_f, ob_g, ob_r, ob_d, cutter]


# --------------------------------------------------------------------------------------
# Light + rotating beams
# --------------------------------------------------------------------------------------
def build_beam_mesh(name, coll, mat, direction):
    n_seg = 28
    x0, x1 = 1.0, BEAM_LEN
    r0 = 0.30
    r1 = r0 + (x1 - x0) * math.tan(math.radians(1.9))
    bm = bmesh.new()
    xs = [x0, x0 + 12.0, x0 + 45.0, x1]
    rings = []
    for x in xs:
        r = r0 + (x - x0) * (r1 - r0) / (x1 - x0)
        rings.append([bm.verts.new((x * direction, r * math.cos(2 * math.pi * j / n_seg),
                                    r * math.sin(2 * math.pi * j / n_seg))) for j in range(n_seg)])
    for i in range(len(rings) - 1):
        for j in range(n_seg):
            j2 = (j + 1) % n_seg
            f = bm.faces.new((rings[i][j], rings[i][j2], rings[i + 1][j2], rings[i + 1][j]))
            f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def build_light(coll, mats_beam):
    pivot = bpy.data.objects.new("LH_BeamPivot", None)
    pivot.empty_display_type = 'PLAIN_AXES'
    pivot.empty_display_size = 1.5
    coll.objects.link(pivot)
    pivot.location = TOWER_ORG + Vector((0, 0, Z_LAMP))
    pivot.rotation_mode = 'XYZ'
    fc = pivot.driver_add('rotation_euler', 2)
    drv = fc.driver
    drv.type = 'SCRIPTED'
    drv.expression = "frame/%d*2*pi" % int(BEAM_PERIOD)
    for m in list(fc.modifiers):
        fc.modifiers.remove(m)

    col = blackbody_rgb(2700.0)
    # point light inside the lantern
    pl = bpy.data.lights.new(PFX + "LampLight", 'POINT')
    pl.energy = 2500.0
    pl.color = col
    pl.shadow_soft_size = 0.12
    plo = bpy.data.objects.new(PFX + "LampLight", pl)
    coll.objects.link(plo)
    plo.parent = pivot
    plo.location = (0, 0, 0)

    objs = [pivot, plo]
    for idx, (sign, name) in enumerate(((1, "A"), (-1, "B"))):
        beam = build_beam_mesh(PFX + "Beam" + name, coll, mats_beam, sign)
        beam.parent = pivot
        beam.visible_shadow = False
        beam.visible_diffuse = False
        beam.visible_glossy = False
        beam.visible_transmission = False
        beam.visible_volume_scatter = False
        objs.append(beam)
        sp = bpy.data.lights.new(PFX + "BeamSpot" + name, 'SPOT')
        sp.energy = 60000.0
        sp.color = col
        sp.spot_size = math.radians(28.0)
        sp.spot_blend = 0.6
        sp.shadow_soft_size = 0.25
        spo = bpy.data.objects.new(PFX + "BeamSpot" + name, sp)
        coll.objects.link(spo)
        spo.parent = pivot
        spo.location = (0, 0, 0)
        spo.rotation_euler = (0.0, math.radians(-90.0 * sign), 0.0)
        objs.append(spo)
    return objs


# --------------------------------------------------------------------------------------
# build()
# --------------------------------------------------------------------------------------
def build():
    cleanup()
    coll = bpy.data.collections.new(COLL_NAME)
    bpy.context.scene.collection.children.link(coll)

    M = dict(rock=mat_rock())
    M['paint'] = mat_tower_paint()
    M['granite'] = mat_granite()
    M['metal'] = mat_metal()
    M['steel'] = mat_steel()
    M['brass'] = mat_brass()
    M['lglass'] = mat_glass_lantern()
    M['lens'] = mat_lens()
    M['core'] = mat_lamp_core()
    M['wdark'] = mat_window_dark()
    M['wlit'] = mat_window_lit()
    M['wood'] = mat_wood_white()
    M['clay'] = mat_clay()
    M['door_red'] = mat_door("DoorRed", (0.30, 0.028, 0.02, 1))
    M['door_green'] = mat_door("DoorGreen", (0.025, 0.09, 0.06, 1))
    M['slate'] = mat_slate()
    M['kstone'] = mat_keeper_stone()
    M['beam'] = mat_beam()

    build_islet(coll, M['rock'])
    build_boulders(coll, M['rock'])
    build_tower(coll, M)
    build_house(coll, M)
    build_light(coll, M['beam'])

    nf = 0
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in coll.all_objects:
        if ob.type == 'MESH' and not ob.hide_render:
            try:
                nf += len(ob.evaluated_get(dg).data.polygons)
            except Exception:
                nf += len(ob.data.polygons)
    print("[lighthouse] built: %d objects, ~%d faces" % (len(coll.all_objects), nf))
    return coll


build()
