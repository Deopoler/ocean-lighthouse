"""Procedural animated flock of realistic herring gulls for Blender 5.3.

Creates an idempotent "Seagulls" collection containing 15 herring gulls sharing
a single procedural mesh, armature data-block, and procedural Cycles materials.
Each gull flaps (~3 Hz with frequency jitter and phase offset) alternating with
gliding phases via built-in driver expressions, and flies along a soaring 3D
arc toward the lighthouse at (10, 80) over frames 1-240 with banking into turns.
"""

import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

COLLECTION_NAME = "Seagulls"
PREFIX = "Gull_"

FRAME_START = 1
FRAME_END = 240
FPS = 24.0

# Camera trajectory from (-20, -80, 3) to (-5, 40, 5) over frames 1-240
CAM_START = Vector((-20.0, -80.0, 3.0))
CAM_END = Vector((-5.0, 40.0, 5.0))
MIN_CAM_CLEARANCE = 5.5  # Strict safety margin > 4.0 m requirement

LIGHTHOUSE_POS = Vector((10.0, 80.0, 0.0))


def srgb_to_linear_channel(c: float) -> float:
    """Convert a single sRGB channel [0..1] to linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def hex_to_rgba(hex_str: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert '#rrggbb' sRGB hex string to linear RGBA tuple."""
    h = hex_str.lstrip("#")
    r = srgb_to_linear_channel(int(h[0:2], 16) / 255.0)
    g = srgb_to_linear_channel(int(h[2:4], 16) / 255.0)
    b = srgb_to_linear_channel(int(h[4:6], 16) / 255.0)
    return (r, g, b, alpha)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep interpolation in [0, 1]."""
    if edge1 == edge0:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def _remove_collection_recursive(col: bpy.types.Collection) -> None:
    """Remove all objects and child collections inside col, then col itself."""
    for child in list(col.children):
        _remove_collection_recursive(child)

    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.data.collections.remove(col)


def cleanup_existing() -> None:
    """Idempotently remove existing Seagulls collection and script-owned data."""
    if bpy.context.mode != "OBJECT" and bpy.context.active_object is not None:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass

    existing_col = bpy.data.collections.get(COLLECTION_NAME)
    if existing_col is not None:
        # Unlink from scene or any parent collections before recursive removal
        for scene in bpy.data.scenes:
            if existing_col.name in scene.collection.children:
                scene.collection.children.unlink(existing_col)
        for parent in bpy.data.collections:
            if existing_col.name in parent.children:
                parent.children.unlink(existing_col)
        _remove_collection_recursive(existing_col)

    # Remove any leftover script-owned objects or data-blocks by prefix
    for obj in list(bpy.data.objects):
        if obj.name.startswith(PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)

    for datablock_store in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.actions,
        bpy.data.curves,
    ):
        for item in list(datablock_store):
            if item.name.startswith(PREFIX):
                datablock_store.remove(item)


def get_or_create_collection() -> bpy.types.Collection:
    """Create the dedicated Seagulls collection linked to the active scene."""
    col = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(col)
    return col


def _init_material(name: str) -> tuple[bpy.types.Material, bpy.types.NodeTree, bpy.types.Node, bpy.types.Node]:
    """Create a material with Principled BSDF and Material Output nodes."""
    mat = bpy.data.materials.new(name=name)
    if getattr(mat, "node_tree", None) is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat, nt, bsdf, out


def _make_mix_rgba(nt: bpy.types.NodeTree, loc: tuple[float, float], fac=0.5, col_a=None, col_b=None):
    """Create a ShaderNodeMix in RGBA mode with Blender 5.x socket indices."""
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.clamp_factor = True
    mix.location = loc
    mix.inputs[0].default_value = fac
    if col_a is not None:
        mix.inputs[6].default_value = col_a
    if col_b is not None:
        mix.inputs[7].default_value = col_b
    return mix


def _make_math(nt: bpy.types.NodeTree, op: str, loc: tuple[float, float], val0=None, val1=None, clamp=False):
    """Create a ShaderNodeMath node."""
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = op
    node.use_clamp = clamp
    node.location = loc
    if val0 is not None:
        node.inputs[0].default_value = val0
    if val1 is not None:
        node.inputs[1].default_value = val1
    return node


def build_materials() -> dict[str, bpy.types.Material]:
    """Build all procedural herring gull materials using shader nodes."""
    mats: dict[str, bpy.types.Material] = {}

    # -------------------------------------------------------------------------
    # 1. Body / Head / Tail White Plumage
    # -------------------------------------------------------------------------
    mat_body, nt, bsdf, _ = _init_material(f"{PREFIX}Mat_Body")

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-850, 0)

    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-650, 0)
    mapping.inputs[3].default_value = (1.0, 0.35, 1.0)  # Stretch feathers along Y
    nt.links.new(tex_coord.outputs[3], mapping.inputs[0])  # Object coords

    noise_macro = nt.nodes.new("ShaderNodeTexNoise")
    noise_macro.location = (-420, 180)
    noise_macro.inputs[2].default_value = 9.0   # Scale
    noise_macro.inputs[3].default_value = 5.0   # Detail
    noise_macro.inputs[4].default_value = 0.6   # Roughness
    nt.links.new(mapping.outputs[0], noise_macro.inputs[0])

    ramp_col = nt.nodes.new("ShaderNodeValToRGB")
    ramp_col.location = (-160, 200)
    ramp_col.color_ramp.elements[0].position = 0.25
    ramp_col.color_ramp.elements[0].color = hex_to_rgba("#d9dcd8")
    ramp_col.color_ramp.elements[1].position = 0.78
    ramp_col.color_ramp.elements[1].color = hex_to_rgba("#f3f4f6")
    nt.links.new(noise_macro.outputs[0], ramp_col.inputs[0])
    nt.links.new(ramp_col.outputs[0], bsdf.inputs["Base Color"])

    noise_feather = nt.nodes.new("ShaderNodeTexNoise")
    noise_feather.location = (-420, -140)
    noise_feather.inputs[2].default_value = 75.0
    noise_feather.inputs[3].default_value = 6.0
    nt.links.new(mapping.outputs[0], noise_feather.inputs[0])

    ramp_rough = nt.nodes.new("ShaderNodeValToRGB")
    ramp_rough.location = (-160, -40)
    ramp_rough.color_ramp.elements[0].position = 0.2
    ramp_rough.color_ramp.elements[0].color = (0.50, 0.50, 0.50, 1.0)
    ramp_rough.color_ramp.elements[1].position = 0.8
    ramp_rough.color_ramp.elements[1].color = (0.72, 0.72, 0.72, 1.0)
    nt.links.new(noise_feather.outputs[0], ramp_rough.inputs[0])
    nt.links.new(ramp_rough.outputs[0], bsdf.inputs["Roughness"])

    bump_body = nt.nodes.new("ShaderNodeBump")
    bump_body.location = (0, -240)
    bump_body.inputs[0].default_value = 0.08
    bump_body.inputs[1].default_value = 0.002
    nt.links.new(noise_feather.outputs[0], bump_body.inputs[2])
    nt.links.new(bump_body.outputs[0], bsdf.inputs["Normal"])

    bsdf.inputs["Subsurface Weight"].default_value = 0.14
    bsdf.inputs["Subsurface Scale"].default_value = 0.015
    bsdf.inputs["Sheen Weight"].default_value = 0.35
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    mats["body"] = mat_body

    # -------------------------------------------------------------------------
    # 2. Wings: Light grey (#9aa5ae) upper, white underwing & trailing edge,
    #    black outer primary wingtips with crisp white mirrors
    # -------------------------------------------------------------------------
    mat_wing, nt, bsdf, _ = _init_material(f"{PREFIX}Mat_Wing")

    uv_main = nt.nodes.new("ShaderNodeUVMap")
    uv_main.uv_map = "UVMap"
    uv_main.location = (-1300, 220)

    sep_uv = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep_uv.location = (-1100, 220)
    nt.links.new(uv_main.outputs[0], sep_uv.inputs[0])
    # sep_uv.outputs[0] = u (span: 0 root -> 1 tip)
    # sep_uv.outputs[1] = v (chord: 0 leading edge -> 1 trailing edge)

    uv_side = nt.nodes.new("ShaderNodeUVMap")
    uv_side.uv_map = "UVSide"
    uv_side.location = (-1300, 460)

    sep_side = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep_side.location = (-1100, 460)
    nt.links.new(uv_side.outputs[0], sep_side.inputs[0])
    # sep_side.outputs[0] = 1.0 on upper surface, 0.0 on lower surface

    # Micro-feather wave + noise
    comb_feather = nt.nodes.new("ShaderNodeCombineXYZ")
    comb_feather.location = (-900, -340)
    nt.links.new(sep_uv.outputs[0], comb_feather.inputs[0])
    nt.links.new(sep_uv.outputs[1], comb_feather.inputs[1])

    wave_feath = nt.nodes.new("ShaderNodeTexWave")
    wave_feath.wave_type = "BANDS"
    wave_feath.bands_direction = "DIAGONAL"
    wave_feath.location = (-700, -320)
    wave_feath.inputs[1].default_value = 22.0  # Scale
    wave_feath.inputs[2].default_value = 2.2   # Distortion
    wave_feath.inputs[3].default_value = 3.0   # Detail
    nt.links.new(comb_feather.outputs[0], wave_feath.inputs[0])

    noise_wing = nt.nodes.new("ShaderNodeTexNoise")
    noise_wing.location = (-700, -540)
    noise_wing.inputs[2].default_value = 65.0
    noise_wing.inputs[3].default_value = 5.0
    nt.links.new(comb_feather.outputs[0], noise_wing.inputs[0])

    # Subtle grey covert variation around #9aa5ae
    ramp_grey = nt.nodes.new("ShaderNodeValToRGB")
    ramp_grey.location = (-440, 480)
    ramp_grey.color_ramp.elements[0].position = 0.25
    ramp_grey.color_ramp.elements[0].color = hex_to_rgba("#8e9aa4")
    ramp_grey.color_ramp.elements[1].position = 0.80
    ramp_grey.color_ramp.elements[1].color = hex_to_rgba("#9aa5ae")
    nt.links.new(wave_feath.outputs[0], ramp_grey.inputs[0])

    # Mix underwing white (#eef0f3) vs upper wing light grey (#9aa5ae)
    # Soft feather blend right at the inner scapular root (u < 0.08)
    ramp_root = nt.nodes.new("ShaderNodeValToRGB")
    ramp_root.location = (-440, 620)
    ramp_root.color_ramp.elements[0].position = 0.02
    ramp_root.color_ramp.elements[0].color = hex_to_rgba("#e4e7ea")
    ramp_root.color_ramp.elements[1].position = 0.09
    ramp_root.color_ramp.elements[1].color = hex_to_rgba("#9aa5ae")
    nt.links.new(sep_uv.outputs[0], ramp_root.inputs[0])

    mix_upper_grey = _make_mix_rgba(nt, (-280, 520), fac=0.75)
    nt.links.new(ramp_grey.outputs[0], mix_upper_grey.inputs[6])
    nt.links.new(ramp_root.outputs[0], mix_upper_grey.inputs[7])

    mix_side = _make_mix_rgba(
        nt,
        (-160, 420),
        col_a=hex_to_rgba("#eef0f3"),
    )
    nt.links.new(sep_side.outputs[0], mix_side.inputs[0])
    nt.links.new(mix_upper_grey.outputs[2], mix_side.inputs[7])

    # White trailing edge along inner/mid upper wing (v > 0.84 and u < 0.72)
    ramp_te_v = nt.nodes.new("ShaderNodeValToRGB")
    ramp_te_v.location = (-700, 320)
    ramp_te_v.color_ramp.elements[0].position = 0.83
    ramp_te_v.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp_te_v.color_ramp.elements[1].position = 0.90
    ramp_te_v.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(sep_uv.outputs[1], ramp_te_v.inputs[0])

    ramp_te_u = nt.nodes.new("ShaderNodeValToRGB")
    ramp_te_u.location = (-700, 120)
    ramp_te_u.color_ramp.elements[0].position = 0.66
    ramp_te_u.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    ramp_te_u.color_ramp.elements[1].position = 0.74
    ramp_te_u.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(sep_uv.outputs[0], ramp_te_u.inputs[0])

    math_te = _make_math(nt, "MULTIPLY", (-420, 240))
    nt.links.new(ramp_te_v.outputs[0], math_te.inputs[0])
    nt.links.new(ramp_te_u.outputs[0], math_te.inputs[1])

    mix_te = _make_mix_rgba(nt, (0, 340), col_b=hex_to_rgba("#f3f4f6"))
    nt.links.new(math_te.outputs[0], mix_te.inputs[0])
    nt.links.new(mix_side.outputs[2], mix_te.inputs[6])

    # Black wingtip mask: u + 0.22 * (1 - v) > 0.76 (diagonal outer primaries)
    sub_v = _make_math(nt, "SUBTRACT", (-900, 40), val0=1.0)
    nt.links.new(sep_uv.outputs[1], sub_v.inputs[1])

    mul_v = _make_math(nt, "MULTIPLY", (-740, 40), val1=0.22)
    nt.links.new(sub_v.outputs[0], mul_v.inputs[0])

    add_tip = _make_math(nt, "ADD", (-560, 40))
    nt.links.new(sep_uv.outputs[0], add_tip.inputs[0])
    nt.links.new(mul_v.outputs[0], add_tip.inputs[1])

    # Tiny noise perturbation on tip boundary for organic feather stepping
    mul_n = _make_math(nt, "MULTIPLY", (-560, -110), val1=0.025)
    nt.links.new(noise_wing.outputs[0], mul_n.inputs[0])

    add_tip_n = _make_math(nt, "ADD", (-400, 20))
    nt.links.new(add_tip.outputs[0], add_tip_n.inputs[0])
    nt.links.new(mul_n.outputs[0], add_tip_n.inputs[1])

    ramp_black = nt.nodes.new("ShaderNodeValToRGB")
    ramp_black.location = (-220, 40)
    ramp_black.color_ramp.elements[0].position = 0.71
    ramp_black.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp_black.color_ramp.elements[1].position = 0.75
    ramp_black.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(add_tip_n.outputs[0], ramp_black.inputs[0])

    # Underside outer primaries are slightly softer dark slate-black than upper
    mix_black_col = _make_mix_rgba(
        nt,
        (-20, 120),
        col_a=hex_to_rgba("#22252a"),
        col_b=hex_to_rgba("#101114"),
    )
    nt.links.new(sep_side.outputs[0], mix_black_col.inputs[0])

    mix_black = _make_mix_rgba(nt, (160, 220))
    nt.links.new(ramp_black.outputs[0], mix_black.inputs[0])
    nt.links.new(mix_te.outputs[2], mix_black.inputs[6])
    nt.links.new(mix_black_col.outputs[2], mix_black.inputs[7])

    # White spots ("mirrors") on outer black wingtips:
    # Spot 1 centered at (u=0.865, v=0.42), Spot 2 at (u=0.928, v=0.50)
    def build_spot_mask(u_c: float, v_c: float, u_r: float, v_r: float, y_off: float):
        du = _make_math(nt, "SUBTRACT", (-900, y_off), val1=u_c)
        nt.links.new(sep_uv.outputs[0], du.inputs[0])
        nu = _make_math(nt, "DIVIDE", (-760, y_off), val1=u_r)
        nt.links.new(du.outputs[0], nu.inputs[0])
        su = _make_math(nt, "POWER", (-620, y_off), val1=2.0)
        nt.links.new(nu.outputs[0], su.inputs[0])

        dv = _make_math(nt, "SUBTRACT", (-900, y_off - 120), val1=v_c)
        nt.links.new(sep_uv.outputs[1], dv.inputs[0])
        nv = _make_math(nt, "DIVIDE", (-760, y_off - 120), val1=v_r)
        nt.links.new(dv.outputs[0], nv.inputs[0])
        sv = _make_math(nt, "POWER", (-620, y_off - 120), val1=2.0)
        nt.links.new(nv.outputs[0], sv.inputs[0])

        rad2 = _make_math(nt, "ADD", (-460, y_off - 60))
        nt.links.new(su.outputs[0], rad2.inputs[0])
        nt.links.new(sv.outputs[0], rad2.inputs[1])

        ramp_sp = nt.nodes.new("ShaderNodeValToRGB")
        ramp_sp.location = (-300, y_off - 60)
        ramp_sp.color_ramp.elements[0].position = 0.65
        ramp_sp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
        ramp_sp.color_ramp.elements[1].position = 1.0
        ramp_sp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
        nt.links.new(rad2.outputs[0], ramp_sp.inputs[0])
        return ramp_sp.outputs[0]

    spot1 = build_spot_mask(0.845, 0.44, 0.028, 0.16, -760)
    spot2 = build_spot_mask(0.915, 0.50, 0.023, 0.14, -1040)

    spots_sum = _make_math(nt, "ADD", (-100, -860), clamp=True)
    nt.links.new(spot1, spots_sum.inputs[0])
    nt.links.new(spot2, spots_sum.inputs[1])

    spots_on_tip = _make_math(nt, "MULTIPLY", (60, -760), clamp=True)
    nt.links.new(spots_sum.outputs[0], spots_on_tip.inputs[0])
    nt.links.new(ramp_black.outputs[0], spots_on_tip.inputs[1])

    mix_spots = _make_mix_rgba(nt, (240, 80), col_b=hex_to_rgba("#f5f6f8"))
    nt.links.new(spots_on_tip.outputs[0], mix_spots.inputs[0])
    nt.links.new(mix_black.outputs[2], mix_spots.inputs[6])
    nt.links.new(mix_spots.outputs[2], bsdf.inputs["Base Color"])

    # Subsurface scattering stronger on white/grey feathers, subtle on black tips
    inv_black = _make_math(nt, "SUBTRACT", (40, -140), val0=1.0)
    nt.links.new(ramp_black.outputs[0], inv_black.inputs[1])
    sss_weight = _make_math(nt, "MULTIPLY", (160, -140), val1=0.16)
    nt.links.new(inv_black.outputs[0], sss_weight.inputs[0])
    nt.links.new(sss_weight.outputs[0], bsdf.inputs["Subsurface Weight"])
    bsdf.inputs["Subsurface Scale"].default_value = 0.012

    # Roughness & feather bump
    ramp_wrough = nt.nodes.new("ShaderNodeValToRGB")
    ramp_wrough.location = (-220, -340)
    ramp_wrough.color_ramp.elements[0].position = 0.25
    ramp_wrough.color_ramp.elements[0].color = (0.48, 0.48, 0.48, 1.0)
    ramp_wrough.color_ramp.elements[1].position = 0.80
    ramp_wrough.color_ramp.elements[1].color = (0.68, 0.68, 0.68, 1.0)
    nt.links.new(wave_feath.outputs[0], ramp_wrough.inputs[0])
    nt.links.new(ramp_wrough.outputs[0], bsdf.inputs["Roughness"])

    bump_wing = nt.nodes.new("ShaderNodeBump")
    bump_wing.location = (60, -380)
    bump_wing.inputs[0].default_value = 0.10
    bump_wing.inputs[1].default_value = 0.0015
    nt.links.new(wave_feath.outputs[0], bump_wing.inputs[2])
    nt.links.new(bump_wing.outputs[0], bsdf.inputs["Normal"])

    bsdf.inputs["Sheen Weight"].default_value = 0.30
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    mats["wing"] = mat_wing

    # -------------------------------------------------------------------------
    # 3. Yellow Beak with Red Gonys Spot
    # -------------------------------------------------------------------------
    mat_beak, nt, bsdf, _ = _init_material(f"{PREFIX}Mat_Beak")

    uv_beak = nt.nodes.new("ShaderNodeUVMap")
    uv_beak.uv_map = "UVMap"
    uv_beak.location = (-950, 120)

    sep_buv = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep_buv.location = (-760, 120)
    nt.links.new(uv_beak.outputs[0], sep_buv.inputs[0])
    # u = beak length [0 base -> 1 tip], v = height [0 bottom -> 1 top]

    noise_beak = nt.nodes.new("ShaderNodeTexNoise")
    noise_beak.location = (-760, -140)
    noise_beak.inputs[2].default_value = 35.0
    noise_beak.inputs[3].default_value = 4.0
    nt.links.new(uv_beak.outputs[0], noise_beak.inputs[0])

    ramp_yellow = nt.nodes.new("ShaderNodeValToRGB")
    ramp_yellow.location = (-440, 220)
    ramp_yellow.color_ramp.elements[0].position = 0.25
    ramp_yellow.color_ramp.elements[0].color = hex_to_rgba("#d49e14")
    ramp_yellow.color_ramp.elements[1].position = 0.80
    ramp_yellow.color_ramp.elements[1].color = hex_to_rgba("#efc024")
    nt.links.new(noise_beak.outputs[0], ramp_yellow.inputs[0])

    # Elliptical red spot on lower mandible near tip (u ~ 0.68, v ~ 0.22)
    b_du = _make_math(nt, "SUBTRACT", (-580, -20), val1=0.68)
    nt.links.new(sep_buv.outputs[0], b_du.inputs[0])
    b_nu = _make_math(nt, "DIVIDE", (-440, -20), val1=0.14)
    nt.links.new(b_du.outputs[0], b_nu.inputs[0])
    b_su = _make_math(nt, "POWER", (-300, -20), val1=2.0)
    nt.links.new(b_nu.outputs[0], b_su.inputs[0])

    b_dv = _make_math(nt, "SUBTRACT", (-580, -160), val1=0.22)
    nt.links.new(sep_buv.outputs[1], b_dv.inputs[0])
    b_nv = _make_math(nt, "DIVIDE", (-440, -160), val1=0.21)
    nt.links.new(b_dv.outputs[0], b_nv.inputs[0])
    b_sv = _make_math(nt, "POWER", (-300, -160), val1=2.0)
    nt.links.new(b_nv.outputs[0], b_sv.inputs[0])

    b_rad = _make_math(nt, "ADD", (-160, -80))
    nt.links.new(b_su.outputs[0], b_rad.inputs[0])
    nt.links.new(b_sv.outputs[0], b_rad.inputs[1])

    b_n_off = _make_math(nt, "MULTIPLY", (-160, -220), val1=0.16)
    nt.links.new(noise_beak.outputs[0], b_n_off.inputs[0])

    b_rad_n = _make_math(nt, "ADD", (-20, -120))
    nt.links.new(b_rad.outputs[0], b_rad_n.inputs[0])
    nt.links.new(b_n_off.outputs[0], b_rad_n.inputs[1])

    ramp_red = nt.nodes.new("ShaderNodeValToRGB")
    ramp_red.location = (80, -80)
    ramp_red.color_ramp.elements[0].position = 0.65
    ramp_red.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    ramp_red.color_ramp.elements[1].position = 1.0
    ramp_red.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(b_rad_n.outputs[0], ramp_red.inputs[0])

    mix_beak = _make_mix_rgba(nt, (120, 140), col_b=hex_to_rgba("#c81e14"))
    nt.links.new(ramp_red.outputs[0], mix_beak.inputs[0])
    nt.links.new(ramp_yellow.outputs[0], mix_beak.inputs[6])
    nt.links.new(mix_beak.outputs[2], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.30
    bsdf.inputs["Subsurface Weight"].default_value = 0.08
    bsdf.inputs["Subsurface Scale"].default_value = 0.008

    bump_beak = nt.nodes.new("ShaderNodeBump")
    bump_beak.location = (60, -280)
    bump_beak.inputs[0].default_value = 0.03
    bump_beak.inputs[1].default_value = 0.001
    nt.links.new(noise_beak.outputs[0], bump_beak.inputs[2])
    nt.links.new(bump_beak.outputs[0], bsdf.inputs["Normal"])
    mats["beak"] = mat_beak

    # -------------------------------------------------------------------------
    # 4. Eyes (Pale yellow iris + glossy dark pupil)
    # -------------------------------------------------------------------------
    mat_eye, nt, bsdf, _ = _init_material(f"{PREFIX}Mat_Eye")

    uv_eye = nt.nodes.new("ShaderNodeUVMap")
    uv_eye.uv_map = "UVMap"
    uv_eye.location = (-720, 0)

    sep_euv = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep_euv.location = (-520, 0)
    nt.links.new(uv_eye.outputs[0], sep_euv.inputs[0])
    # u = radial distance from pupil center [0 center -> 1 rim]

    ramp_eye = nt.nodes.new("ShaderNodeValToRGB")
    ramp_eye.location = (-260, 0)
    ramp_eye.color_ramp.elements[0].position = 0.58
    ramp_eye.color_ramp.elements[0].color = hex_to_rgba("#040405")
    ramp_eye.color_ramp.elements[1].position = 0.66
    ramp_eye.color_ramp.elements[1].color = hex_to_rgba("#c8b256")
    nt.links.new(sep_euv.outputs[0], ramp_eye.inputs[0])
    nt.links.new(ramp_eye.outputs[0], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.04
    bsdf.inputs["Coat Weight"].default_value = 0.85
    bsdf.inputs["Coat Roughness"].default_value = 0.03
    mats["eye"] = mat_eye

    # -------------------------------------------------------------------------
    # 5. Tucked Legs / Feet (Pinkish-flesh with fine scale voronoi bump)
    # -------------------------------------------------------------------------
    mat_legs, nt, bsdf, _ = _init_material(f"{PREFIX}Mat_Legs")

    tc_legs = nt.nodes.new("ShaderNodeTexCoord")
    tc_legs.location = (-600, 0)
    vor_legs = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_legs.location = (-380, -80)
    vor_legs.inputs[2].default_value = 180.0
    nt.links.new(tc_legs.outputs[3], vor_legs.inputs[0])

    bsdf.inputs["Base Color"].default_value = hex_to_rgba("#d48a82")
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Subsurface Weight"].default_value = 0.10
    bsdf.inputs["Subsurface Scale"].default_value = 0.008

    bump_legs = nt.nodes.new("ShaderNodeBump")
    bump_legs.location = (-100, -120)
    bump_legs.inputs[0].default_value = 0.08
    bump_legs.inputs[1].default_value = 0.001
    nt.links.new(vor_legs.outputs[0], bump_legs.inputs[2])
    nt.links.new(bump_legs.outputs[0], bsdf.inputs["Normal"])
    mats["legs"] = mat_legs

    return mats


def _assign_vertex_weights(mesh: bpy.types.Mesh, obj: bpy.types.Object, populate_mesh: bool = True) -> None:
    """Assign smooth skinning weights for body, shoulder.L/R, and elbow.L/R."""
    vg_body = obj.vertex_groups.new(name="body")
    vg_sh_l = obj.vertex_groups.new(name="shoulder.L")
    vg_el_l = obj.vertex_groups.new(name="elbow.L")
    vg_sh_r = obj.vertex_groups.new(name="shoulder.R")
    vg_el_r = obj.vertex_groups.new(name="elbow.R")

    if not populate_mesh:
        return

    # Classify vertices: wings have |x| > 0.032 and -0.18 <= y <= +0.14 and z > -0.025
    # (Body/head/beak/tail/legs stay strictly on the body bone).
    for v in mesh.vertices:
        co = v.co
        ax = abs(co.x)
        is_wing = (
            ax > 0.034
            and (-0.175 <= co.y <= 0.135)
            and (-0.022 <= co.z <= 0.085)
            and not (co.y < -0.135 and ax < 0.085)  # Tail stays on body
        )

        if not is_wing:
            vg_body.add([v.index], 1.0, "REPLACE")
            continue

        is_left = co.x < 0.0
        vg_sh = vg_sh_l if is_left else vg_sh_r
        vg_el = vg_el_l if is_left else vg_el_r

        # Smooth blend from body -> shoulder -> elbow along span
        if ax < 0.105:
            t_sh = smoothstep(0.038, 0.105, ax)
            if 1.0 - t_sh > 1e-4:
                vg_body.add([v.index], 1.0 - t_sh, "REPLACE")
            if t_sh > 1e-4:
                vg_sh.add([v.index], t_sh, "REPLACE")
        elif ax < 0.245:
            vg_sh.add([v.index], 1.0, "REPLACE")
        elif ax < 0.365:
            t_el = smoothstep(0.245, 0.365, ax)
            if 1.0 - t_el > 1e-4:
                vg_sh.add([v.index], 1.0 - t_el, "REPLACE")
            if t_el > 1e-4:
                vg_el.add([v.index], t_el, "REPLACE")
        else:
            vg_el.add([v.index], 1.0, "REPLACE")


def build_gull_mesh(mats: dict[str, bpy.types.Material]) -> bpy.types.Mesh:
    """Build procedural herring gull mesh (~0.60 m long, 1.40 m wingspan)."""
    bm = bmesh.new()
    uv_main = bm.loops.layers.uv.new("UVMap")
    uv_side = bm.loops.layers.uv.new("UVSide")

    MAT_BODY = 0
    MAT_WING = 1
    MAT_BEAK = 2
    MAT_EYE = 3
    MAT_LEGS = 4

    # -------------------------------------------------------------------------
    # 1. Fuselage Loft: Rump -> Body -> Neck -> Head -> Beak
    # -------------------------------------------------------------------------
    n_ring = 10
    # (y, z_center, rx, rz_top, rz_bot)
    stations = [
        (-0.155, 0.004, 0.026, 0.011, 0.013),   # 0: rump / tail base
        (-0.115, 0.002, 0.043, 0.027, 0.031),   # 1: rear body
        (-0.055, 0.000, 0.056, 0.041, 0.047),   # 2: mid body
        ( 0.015, -0.002, 0.062, 0.048, 0.058),  # 3: full breast
        ( 0.080, 0.004, 0.055, 0.044, 0.051),   # 4: front chest
        ( 0.135, 0.018, 0.037, 0.033, 0.035),   # 5: lower neck
        ( 0.175, 0.030, 0.030, 0.027, 0.027),   # 6: mid neck
        ( 0.205, 0.040, 0.030, 0.031, 0.024),   # 7: head back
        ( 0.232, 0.042, 0.026, 0.030, 0.021),   # 8: crown / eyes
        ( 0.252, 0.038, 0.016, 0.019, 0.015),   # 9: forehead / face
        ( 0.258, 0.034, 0.011, 0.012, 0.011),   # 10: beak base (mat boundary)
        ( 0.278, 0.032, 0.0082, 0.0102, 0.0098),# 11: mid beak
        ( 0.298, 0.030, 0.0062, 0.0088, 0.0105),# 12: gonys bulge (red spot)
        ( 0.312, 0.026, 0.0038, 0.0062, 0.0048),# 13: hooked culmen
        ( 0.321, 0.020, 0.0014, 0.0022, 0.0018),# 14: beak tip
    ]

    rings = []
    for y, zc, rx, rzt, rzb in stations:
        ring = []
        for k in range(n_ring):
            ang = 2.0 * math.pi * k / n_ring
            sx = math.sin(ang)
            cz = math.cos(ang)
            rz = rzt if cz >= 0.0 else rzb
            # Slight plump flattening on belly
            v = bm.verts.new((rx * sx, y, zc + rz * cz))
            ring.append(v)
        rings.append(ring)

    beak_start_idx = 10
    beak_y0 = stations[beak_start_idx][0]
    beak_y1 = stations[-1][0]

    for i in range(len(rings) - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        is_beak = i >= beak_start_idx
        mat_idx = MAT_BEAK if is_beak else MAT_BODY

        for k in range(n_ring):
            kn = (k + 1) % n_ring
            f = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
            f.material_index = mat_idx
            f.smooth = True
            if is_beak:
                for loop in f.loops:
                    co = loop.vert.co
                    u = max(0.0, min(1.0, (co.y - beak_y0) / (beak_y1 - beak_y0)))
                    v_norm = max(0.0, min(1.0, (co.z - 0.018) / (0.046 - 0.018)))
                    loop[uv_main].uv = (u, v_norm)
                    loop[uv_side].uv = (1.0, 0.5)

    # Cap rump and beak tip cleanly with quads/fan
    f_rump = bm.faces.new(list(reversed(rings[0])))
    f_rump.material_index = MAT_BODY
    f_rump.smooth = True

    f_tip = bm.faces.new(rings[-1])
    f_tip.material_index = MAT_BEAK
    f_tip.smooth = True
    for loop in f_tip.loops:
        loop[uv_main].uv = (1.0, 0.5)

    # -------------------------------------------------------------------------
    # 2. Fan-Shaped Tail
    # -------------------------------------------------------------------------
    tail_span = 4
    tail_chord = 3
    tail_top_grid = []
    tail_bot_grid = []
    for j in range(tail_chord):
        tv = j / (tail_chord - 1)  # 0 root -> 1 tip
        row_t = []
        row_b = []
        half_w = lerp(0.028, 0.078, tv)
        thick = lerp(0.010, 0.0025, tv)
        for i in range(tail_span):
            tu = (i / (tail_span - 1)) * 2.0 - 1.0  # -1..1
            x = tu * half_w
            # Rounded fan tip (center rectrices slightly longer than outer)
            y_end = -0.282 + 0.022 * (tu * tu)
            y = lerp(-0.142, y_end, tv)
            z_mid = lerp(0.004, -0.003, tv) + 0.003 * (1.0 - tu * tu)
            row_t.append(bm.verts.new((x, y, z_mid + 0.5 * thick)))
            row_b.append(bm.verts.new((x, y, z_mid - 0.5 * thick)))
        tail_top_grid.append(row_t)
        tail_bot_grid.append(row_b)

    for j in range(tail_chord - 1):
        for i in range(tail_span - 1):
            ft = bm.faces.new((
                tail_top_grid[j][i],
                tail_top_grid[j + 1][i],
                tail_top_grid[j + 1][i + 1],
                tail_top_grid[j][i + 1],
            ))
            ft.material_index = MAT_BODY
            ft.smooth = True
            fb = bm.faces.new((
                tail_bot_grid[j][i],
                tail_bot_grid[j][i + 1],
                tail_bot_grid[j + 1][i + 1],
                tail_bot_grid[j + 1][i],
            ))
            fb.material_index = MAT_BODY
            fb.smooth = True

    # Stitch tail trailing edge and side edges
    for i in range(tail_span - 1):
        fe = bm.faces.new((
            tail_top_grid[-1][i],
            tail_bot_grid[-1][i],
            tail_bot_grid[-1][i + 1],
            tail_top_grid[-1][i + 1],
        ))
        fe.material_index = MAT_BODY
        fe.smooth = True
    for j in range(tail_chord - 1):
        fl = bm.faces.new((
            tail_top_grid[j][0],
            tail_bot_grid[j][0],
            tail_bot_grid[j + 1][0],
            tail_top_grid[j + 1][0],
        ))
        fl.material_index = MAT_BODY
        fl.smooth = True
        fr = bm.faces.new((
            tail_top_grid[j][-1],
            tail_top_grid[j + 1][-1],
            tail_bot_grid[j + 1][-1],
            tail_bot_grid[j][-1],
        ))
        fr.material_index = MAT_BODY
        fr.smooth = True

    # -------------------------------------------------------------------------
    # 3. Tapered Gull Wings (1.40 m wingspan, airfoil cross-section)
    # -------------------------------------------------------------------------
    # (abs_x, y_le, y_te, z_mid, thickness)
    wing_stations = [
        (0.038,  0.092, -0.088, 0.018, 0.025),  # root tucked into torso
        (0.085,  0.100, -0.085, 0.024, 0.021),  # shoulder
        (0.190,  0.112, -0.068, 0.036, 0.016),  # mid inner wing
        (0.305,  0.122, -0.048, 0.046, 0.0125), # carpal / wrist kink
        (0.435,  0.072, -0.068, 0.034, 0.0095), # mid outer primaries
        (0.565, -0.005, -0.105, 0.016, 0.0068), # outer primaries
        (0.655, -0.080, -0.140, 0.004, 0.0042), # near tip
        (0.700, -0.132, -0.160, -0.002, 0.0028),# wingtip (total span 1.40 m)
    ]

    # Airfoil profile: (chord_frac, z_offset_factor, is_upper_flag)
    # 8 points around airfoil: LE(0) -> upper(1,2,3) -> TE(4) -> lower(5,6,7)
    airfoil_pts = [
        (0.00, -0.05),  # 0: leading edge
        (0.25,  0.62),  # 1: upper front camber peak
        (0.55,  0.46),  # 2: upper mid
        (0.82,  0.22),  # 3: upper rear
        (1.00, -0.04),  # 4: trailing edge
        (0.82, -0.14),  # 5: lower rear
        (0.55, -0.26),  # 6: lower mid
        (0.25, -0.34),  # 7: lower front
    ]
    n_af = len(airfoil_pts)

    x_min_w = wing_stations[0][0]
    x_max_w = wing_stations[-1][0]

    for sign in (-1.0, 1.0):
        station_rings = []
        station_uvs = []
        for ax, y_le, y_te, z_mid, thick in wing_stations:
            ring = []
            uvs = []
            u_span = (ax - x_min_w) / (x_max_w - x_min_w)
            for cf, zf in airfoil_pts:
                y = lerp(y_le, y_te, cf)
                z = z_mid + zf * thick
                v = bm.verts.new((sign * ax, y, z))
                ring.append(v)
                uvs.append((u_span, cf))
            station_rings.append(ring)
            station_uvs.append(uvs)

        for s_idx in range(len(station_rings) - 1):
            r0 = station_rings[s_idx]
            r1 = station_rings[s_idx + 1]
            uv0 = station_uvs[s_idx]
            uv1 = station_uvs[s_idx + 1]

            for k in range(n_af):
                kn = (k + 1) % n_af
                # Upper faces are k in {0, 1, 2, 3}; lower faces are k in {4, 5, 6, 7}
                is_upper = k < 4
                side_val = 1.0 if is_upper else 0.0

                if sign > 0.0:
                    verts = (r0[k], r1[k], r1[kn], r0[kn])
                    uvs_f = (uv0[k], uv1[k], uv1[kn], uv0[kn])
                else:
                    verts = (r0[k], r0[kn], r1[kn], r1[k])
                    uvs_f = (uv0[k], uv0[kn], uv1[kn], uv1[k])

                f = bm.faces.new(verts)
                f.material_index = MAT_WING
                f.smooth = True
                for loop, uv_val in zip(f.loops, uvs_f):
                    loop[uv_main].uv = uv_val
                    loop[uv_side].uv = (side_val, 0.5)

        # Cap wingtip cleanly with quads across upper-to-lower pairs
        tip_r = station_rings[-1]
        tip_uv = station_uvs[-1]
        tip_quads = [
            (0, 1, 7),  # tri at leading tip
            (1, 2, 6, 7),
            (2, 3, 5, 6),
            (3, 4, 5),  # tri at trailing tip
        ]
        for q in tip_quads:
            if sign > 0.0:
                idxs = q
            else:
                idxs = tuple(reversed(q))
            f = bm.faces.new([tip_r[idx] for idx in idxs])
            f.material_index = MAT_WING
            f.smooth = True
            for loop, idx in zip(f.loops, idxs):
                loop[uv_main].uv = tip_uv[idx]
                loop[uv_side].uv = (1.0 if idx <= 4 else 0.0, 0.5)

    # -------------------------------------------------------------------------
    # 4. Eyes (Left & Right)
    # -------------------------------------------------------------------------
    for sign in (-1.0, 1.0):
        res = bmesh.ops.create_uvsphere(
            bm,
            u_segments=6,
            v_segments=4,
            radius=0.0060,
            matrix=Matrix.Translation((sign * 0.0225, 0.226, 0.044))
            @ Matrix.Scale(0.65, 4, (1.0, 0.0, 0.0)),
        )
        eye_verts = set(res["verts"])
        eye_center = Vector((sign * 0.0225, 0.226, 0.044))
        for f in bm.faces:
            if all(v in eye_verts for v in f.verts):
                f.material_index = MAT_EYE
                f.smooth = True
                for loop in f.loops:
                    d = loop.vert.co - eye_center
                    # Outward facing direction is (sign, 0.15, 0)
                    # Map radial angle from outward axis into [0..1]
                    proj = Vector((0.0, d.y, d.z)).length / 0.0060
                    out_dot = d.x * sign
                    u_eye = min(1.0, proj) if out_dot > 0 else 1.0
                    loop[uv_main].uv = (u_eye, 0.5)

    # -------------------------------------------------------------------------
    # 5. Tucked Legs & Webbed Feet
    # -------------------------------------------------------------------------
    for sign in (-1.0, 1.0):
        leg_stations = [
            (sign * 0.016, -0.055, -0.032, 0.0050, 0.0045),
            (sign * 0.017, -0.105, -0.025, 0.0042, 0.0040),
            (sign * 0.018, -0.145, -0.016, 0.0038, 0.0036),
            (sign * 0.019, -0.165, -0.012, 0.0075, 0.0028),  # folded web foot
            (sign * 0.020, -0.182, -0.009, 0.0095, 0.0020),  # webbed foot tip
        ]
        leg_rings = []
        for lx, ly, lz, rlx, rlz in leg_stations:
            ring = []
            for k in range(4):
                ang = (math.pi / 4.0) + k * (math.pi / 2.0)
                ring.append(bm.verts.new((lx + rlx * math.cos(ang), ly, lz + rlz * math.sin(ang))))
            leg_rings.append(ring)

        for i in range(len(leg_rings) - 1):
            r0 = leg_rings[i]
            r1 = leg_rings[i + 1]
            for k in range(4):
                kn = (k + 1) % 4
                f = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
                f.material_index = MAT_LEGS
                f.smooth = True
        f_foot_tip = bm.faces.new(leg_rings[-1])
        f_foot_tip.material_index = MAT_LEGS
        f_foot_tip.smooth = True

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name=f"{PREFIX}Mesh")
    bm.to_mesh(mesh)
    bm.free()

    mesh.materials.append(mats["body"])
    mesh.materials.append(mats["wing"])
    mesh.materials.append(mats["beak"])
    mesh.materials.append(mats["eye"])
    mesh.materials.append(mats["legs"])

    return mesh


def build_gull_armature_data(col: bpy.types.Collection) -> bpy.types.Armature:
    """Create shared Armature data-block with body, shoulder.L/R, elbow.L/R."""
    arm_data = bpy.data.armatures.new(name=f"{PREFIX}ArmatureData")
    tmp_obj = bpy.data.objects.new(name=f"{PREFIX}TmpRig", object_data=arm_data)
    col.objects.link(tmp_obj)

    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = tmp_obj
    bpy.ops.object.mode_set(mode="EDIT")

    eb = arm_data.edit_bones
    body_b = eb.new("body")
    body_b.head = (0.0, -0.05, 0.0)
    body_b.tail = (0.0, 0.10, 0.0)

    for suffix, sign in (("L", -1.0), ("R", 1.0)):
        sh = eb.new(f"shoulder.{suffix}")
        sh.head = (sign * 0.048, 0.015, 0.020)
        sh.tail = (sign * 0.305, 0.042, 0.046)
        sh.parent = body_b
        sh.align_roll(Vector((0.0, 0.0, 1.0)))

        el = eb.new(f"elbow.{suffix}")
        el.head = (sign * 0.305, 0.042, 0.046)
        el.tail = (sign * 0.700, -0.142, -0.002)
        el.parent = sh
        el.use_connect = True
        el.align_roll(Vector((0.0, 0.0, 1.0)))

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(tmp_obj, do_unlink=True)
    if prev_active is not None and prev_active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = prev_active

    return arm_data


def _add_env_var(driver: bpy.types.Driver, rig_obj: bpy.types.Object) -> None:
    """Add driver variable 'e' referencing rig_obj['flap_env']."""
    var = driver.variables.new()
    var.name = "e"
    var.type = "SINGLE_PROP"
    target = var.targets[0]
    target.id_type = "OBJECT"
    target.id = rig_obj
    target.data_path = '["flap_env"]'


def add_wing_flap_drivers(rig_obj: bpy.types.Object, rng: random.Random, glide_bias: float) -> None:
    """Attach built-in driver expressions for ~3 Hz flapping + glide phases."""
    # ~3 flaps/sec with per-bird jitter (2.73 - 3.27 Hz)
    flap_hz = 3.0 * rng.uniform(0.91, 1.09)
    w_flap = 2.0 * math.pi * flap_hz / FPS
    phase = rng.uniform(0.0, 2.0 * math.pi)

    # Slow alternation between flapping bursts (~2 s) and glide phases (~2-3 s)
    glide_hz = rng.uniform(0.18, 0.26)
    w_glide = 2.0 * math.pi * glide_hz / FPS
    glide_phase = rng.uniform(0.0, 2.0 * math.pi)

    # Store flap/glide envelope [0, 1] on a custom property so expressions stay short
    rig_obj["flap_env"] = 0.0
    d_env = rig_obj.driver_add('["flap_env"]').driver
    d_env.type = "SCRIPTED"
    d_env.expression = (
        f"max(0.0, min(1.0, 2.2 * sin(frame * {w_glide:.5f} + {glide_phase:.4f}) "
        f"+ {glide_bias:.4f}))"
    )

    p_expr = f"(frame * {w_flap:.5f} + {phase:.4f})"

    # Shoulder flap: positive local X rotates wingtip UP on both L and R
    sh_flap_expr = (
        f"(1 - e) * 0.10 + e * (0.04 + 0.42 * (sin({p_expr}) + 0.18 * sin(2 * {p_expr} - 0.5)))"
    )
    # Elbow flap: lags shoulder by ~1.05 rad; holds gull-wing anhedral (-0.22 rad) in glide
    el_flap_expr = (
        f"(1 - e) * -0.22 + e * (-0.08 + 0.48 * sin({p_expr} - 1.05))"
    )

    for suffix, sign in (("L", -1.0), ("R", 1.0)):
        pb_sh = rig_obj.pose.bones[f"shoulder.{suffix}"]
        pb_el = rig_obj.pose.bones[f"elbow.{suffix}"]
        pb_sh.rotation_mode = "XYZ"
        pb_el.rotation_mode = "XYZ"

        # Flap up/down (local X)
        d_sh_x = pb_sh.driver_add("rotation_euler", 0).driver
        d_sh_x.type = "SCRIPTED"
        _add_env_var(d_sh_x, rig_obj)
        d_sh_x.expression = sh_flap_expr

        d_el_x = pb_el.driver_add("rotation_euler", 0).driver
        d_el_x.type = "SCRIPTED"
        _add_env_var(d_el_x, rig_obj)
        d_el_x.expression = el_flap_expr

        # Subtle feathering twist around span axis (local Y, opposite sign L/R)
        d_sh_y = pb_sh.driver_add("rotation_euler", 1).driver
        d_sh_y.type = "SCRIPTED"
        _add_env_var(d_sh_y, rig_obj)
        d_sh_y.expression = f"{-sign:.1f} * 0.11 * e * cos({p_expr})"

        # Slight wing sweep back during glide & upstroke (local Z, opposite sign L/R)
        d_el_z = pb_el.driver_add("rotation_euler", 2).driver
        d_el_z.type = "SCRIPTED"
        _add_env_var(d_el_z, rig_obj)
        d_el_z.expression = (
            f"{sign:.1f} * ((1 - e) * -0.07 + e * 0.08 * cos({p_expr}))"
        )

    # Subtle body counter-heave and pitch nod during flapping
    pb_body = rig_obj.pose.bones["body"]
    pb_body.rotation_mode = "XYZ"

    d_bz = pb_body.driver_add("location", 2).driver
    d_bz.type = "SCRIPTED"
    _add_env_var(d_bz, rig_obj)
    d_bz.expression = f"e * -0.018 * sin({p_expr})"

    d_bx = pb_body.driver_add("rotation_euler", 0).driver
    d_bx.type = "SCRIPTED"
    _add_env_var(d_bx, rig_obj)
    d_bx.expression = f"e * 0.035 * cos({p_expr})"


def _cam_pos_at(u: float) -> Vector:
    """Linearly interpolate camera path position for u in [0, 1]."""
    return CAM_START.lerp(CAM_END, u)


def _dist_point_to_segment(pt: Vector, a: Vector, b: Vector) -> float:
    """Minimum 3D distance from pt to line segment [a, b]."""
    ab = b - a
    denom = ab.dot(ab)
    if denom < 1e-9:
        return (pt - a).length
    t = max(0.0, min(1.0, (pt - a).dot(ab) / denom))
    proj = a + ab * t
    return (pt - proj).length


def _generate_flock_specs() -> list[dict]:
    """Define 15 organic gull flight specifications from y~-40 toward (10,80)."""
    # Each spec defines start/end positions, lateral soaring arc amplitude/freq,
    # altitude thermal wave, and glide bias.
    # Birds 0-3 pass fairly near the camera path (6-11 m) while staying >= 5.5 m away.
    specs = [
        # --- Hero / Foreground gulls passing near camera path ---
        {
            "p0": Vector((-25.0, -40.0, 8.4)),
            "p1": Vector((2.0, 50.0, 11.4)),
            "arc_x": 5.5, "arc_y": 0.0, "arc_z": 1.6, "cycles": 1.15, "phase": 0.3,
            "glide_bias": 0.15,
        },
        {
            "p0": Vector((-9.0, -42.0, 7.6)),
            "p1": Vector((-1.0, 47.0, 9.8)),
            "arc_x": -4.2, "arc_y": 0.0, "arc_z": 1.4, "cycles": 1.25, "phase": 1.8,
            "glide_bias": 0.10,
        },
        {
            "p0": Vector((-22.0, -34.0, 9.6)),
            "p1": Vector((5.0, 58.0, 12.8)),
            "arc_x": 4.8, "arc_y": 0.0, "arc_z": 1.8, "cycles": 1.05, "phase": 4.1,
            "glide_bias": 0.20,
        },
        {
            "p0": Vector((-14.0, -36.0, 11.4)),
            "p1": Vector((14.0, 64.0, 14.8)),
            "arc_x": -5.0, "arc_y": 0.0, "arc_z": 2.0, "cycles": 0.95, "phase": 2.7,
            "glide_bias": 0.18,
        },
        # --- Mid-altitude loose flock heading toward lighthouse islet ---
        {
            "p0": Vector((-4.0, -42.0, 12.8)),
            "p1": Vector((18.0, 86.0, 16.5)),
            "arc_x": 6.5, "arc_y": 0.0, "arc_z": 2.2, "cycles": 1.10, "phase": 0.9,
            "glide_bias": 0.12,
        },
        {
            "p0": Vector((4.0, -38.0, 10.5)),
            "p1": Vector((24.0, 82.0, 14.2)),
            "arc_x": -5.8, "arc_y": 0.0, "arc_z": 1.7, "cycles": 1.30, "phase": 3.4,
            "glide_bias": 0.08,
        },
        {
            "p0": Vector((12.0, -45.0, 13.5)),
            "p1": Vector((2.0, 90.0, 17.8)),
            "arc_x": 7.0, "arc_y": 0.0, "arc_z": 2.4, "cycles": 1.00, "phase": 5.0,
            "glide_bias": 0.16,
        },
        {
            "p0": Vector((-18.0, -40.0, 15.2)),
            "p1": Vector((26.0, 74.0, 18.4)),
            "arc_x": -6.2, "arc_y": 0.0, "arc_z": 2.1, "cycles": 0.90, "phase": 1.2,
            "glide_bias": 0.05,
        },
        {
            "p0": Vector((18.0, -34.0, 11.8)),
            "p1": Vector((28.0, 88.0, 15.6)),
            "arc_x": -6.8, "arc_y": 0.0, "arc_z": 1.9, "cycles": 1.15, "phase": 2.1,
            "glide_bias": 0.10,
        },
        {
            "p0": Vector((0.0, -30.0, 16.0)),
            "p1": Vector((-4.0, 92.0, 19.6)),
            "arc_x": 8.0, "arc_y": 0.0, "arc_z": 2.3, "cycles": 0.85, "phase": 4.6,
            "glide_bias": 0.02,
        },
        # --- Higher soaring birds wheeling around and past the lighthouse ---
        {
            "p0": Vector((-12.0, -44.0, 18.5)),
            "p1": Vector((16.0, 94.0, 22.8)),
            "arc_x": 9.5, "arc_y": 0.0, "arc_z": 2.5, "cycles": 0.90, "phase": 0.6,
            "glide_bias": -0.05,
        },
        {
            "p0": Vector((8.0, -40.0, 20.0)),
            "p1": Vector((-6.0, 86.0, 23.2)),
            "arc_x": -8.5, "arc_y": 0.0, "arc_z": 2.0, "cycles": 1.00, "phase": 3.8,
            "glide_bias": -0.08,
        },
        {
            "p0": Vector((24.0, -42.0, 17.2)),
            "p1": Vector((22.0, 95.0, 21.0)),
            "arc_x": -9.0, "arc_y": 0.0, "arc_z": 2.6, "cycles": 0.80, "phase": 2.4,
            "glide_bias": -0.04,
        },
        {
            "p0": Vector((-26.0, -35.0, 19.5)),
            "p1": Vector((30.0, 80.0, 22.4)),
            "arc_x": 7.5, "arc_y": 0.0, "arc_z": 2.2, "cycles": 0.95, "phase": 5.4,
            "glide_bias": -0.02,
        },
        {
            "p0": Vector((15.0, -28.0, 21.5)),
            "p1": Vector((6.0, 96.0, 24.0)),
            "arc_x": 8.0, "arc_y": 0.0, "arc_z": 1.5, "cycles": 0.85, "phase": 1.6,
            "glide_bias": -0.10,
        },
    ]
    return specs


def _eval_trajectory(spec: dict, u: float) -> Vector:
    """Evaluate smooth soaring 3D position at normalized time u in [0, 1]."""
    base = spec["p0"].lerp(spec["p1"], u)
    # Envelope tapers at ends slightly while allowing smooth mid-flight soaring curves
    theta = 2.0 * math.pi * spec["cycles"] * u + spec["phase"]
    dx = spec["arc_x"] * math.sin(theta)
    dz = spec["arc_z"] * math.sin(theta * 1.3 + 0.7)
    pos = Vector((base.x + dx, base.y, base.z + dz))
    # Clamp strictly within 6.0 - 25.0 m altitude
    pos.z = max(6.0, min(25.0, pos.z))
    return pos


def _verify_and_adjust_paths(specs: list[dict]) -> None:
    """Ensure camera clearance (>= 5.5 m), lighthouse clearance, and bird spacing."""
    # Sample across all 240 frames
    n_samples = FRAME_END - FRAME_START + 1
    for idx, spec in enumerate(specs):
        # Push away gently if any point comes within MIN_CAM_CLEARANCE of camera path
        for _ in range(8):
            min_seg_d = 999.0
            worst_u = 0.0
            for f in range(n_samples):
                u = f / (n_samples - 1)
                pos = _eval_trajectory(spec, u)
                d_seg = _dist_point_to_segment(pos, CAM_START, CAM_END)
                if d_seg < min_seg_d:
                    min_seg_d = d_seg
                    worst_u = u
            if min_seg_d >= MIN_CAM_CLEARANCE:
                break
            # Raise altitude slightly to preserve lateral trajectory while clearing camera
            spec["p0"].z = min(23.0, spec["p0"].z + 1.0)
            spec["p1"].z = min(23.0, spec["p1"].z + 1.0)

        # Keep lateral clearance from lighthouse tower core at (10, 80) if below z=22
        end_pos = _eval_trajectory(spec, 1.0)
        xy_dist_lh = math.hypot(end_pos.x - LIGHTHOUSE_POS.x, end_pos.y - LIGHTHOUSE_POS.y)
        if xy_dist_lh < 6.0 and end_pos.z < 22.0:
            sign_x = 1.0 if end_pos.x >= LIGHTHOUSE_POS.x else -1.0
            spec["p1"].x += sign_x * (6.5 - xy_dist_lh)

    # Ensure minimum inter-bird spacing (>= 2.5 m) across all frames
    for _ in range(12):
        adjusted = False
        for f in range(0, n_samples, 3):
            u = f / (n_samples - 1)
            pts = [_eval_trajectory(s, u) for s in specs]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = (pts[i] - pts[j]).length
                    if d < 2.6:
                        # Nudge bird j away horizontally/vertically
                        dx = 1.0 if (pts[j].x >= pts[i].x) else -1.0
                        specs[j]["p0"].x += dx * 1.4
                        specs[j]["p1"].x += dx * 1.4
                        specs[j]["p0"].z = min(23.5, max(6.5, specs[j]["p0"].z + 0.8))
                        specs[j]["p1"].z = min(23.5, max(6.5, specs[j]["p1"].z + 0.8))
                        adjusted = True
        if not adjusted:
            break


def animate_flight_path(rig_obj: bpy.types.Object, spec: dict) -> None:
    """Keyframe location and curvature-banked orientation over frames 1-240."""
    rig_obj.rotation_mode = "XYZ"
    world_up = Vector((0.0, 0.0, 1.0))

    prev_euler = None
    # Keyframe every 4 frames plus final frame 240 for smooth Bezier interpolation
    frames = list(range(FRAME_START, FRAME_END, 4)) + [FRAME_END]
    dt = 1.0 / FPS
    du_eps = 0.004

    for frame in frames:
        u = (frame - FRAME_START) / (FRAME_END - FRAME_START)
        u_prev = max(0.0, u - du_eps)
        u_next = min(1.0, u + du_eps)

        p = _eval_trajectory(spec, u)
        p_prev = _eval_trajectory(spec, u_prev)
        p_next = _eval_trajectory(spec, u_next)

        total_t = (FRAME_END - FRAME_START) * dt
        v_prev = (p - p_prev) / ((u - u_prev) * total_t) if u > u_prev else (p_next - p) / ((u_next - u) * total_t)
        v_next = (p_next - p) / ((u_next - u) * total_t) if u_next > u else v_prev
        vel = 0.5 * (v_prev + v_next)
        acc = (v_next - v_prev) / (0.5 * (u_next - u_prev) * total_t) if u_next > u_prev else Vector((0.0, 0.0, 0.0))

        fwd = vel.normalized()
        right0 = fwd.cross(world_up).normalized()
        up0 = right0.cross(fwd).normalized()

        # Coordinated turn bank angle: bank inward toward turn center
        a_lat = acc.dot(right0)
        bank_angle = max(-0.56, min(0.56, math.atan(a_lat / 9.81) * 1.45))

        up_banked = (up0 * math.cos(bank_angle) + right0 * math.sin(bank_angle)).normalized()
        right_banked = fwd.cross(up_banked).normalized()

        rot_mat = Matrix((right_banked, fwd, up_banked)).transposed()
        if prev_euler is None:
            euler = rot_mat.to_euler("XYZ")
        else:
            euler = rot_mat.to_euler("XYZ", prev_euler)
        prev_euler = euler

        rig_obj.location = p
        rig_obj.rotation_euler = euler
        rig_obj.keyframe_insert(data_path="location", frame=frame)
        rig_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set smooth Bezier interpolation via Blender 5.x layered Action API
    anim = rig_obj.animation_data
    if anim is not None and anim.action is not None:
        act = anim.action
        if len(act.layers) > 0 and len(act.layers[0].strips) > 0 and len(act.slots) > 0:
            cb = act.layers[0].strips[0].channelbag(act.slots[0])
            for fc in cb.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"


def build() -> bpy.types.Collection:
    """Build the complete animated Seagulls collection idempotently."""
    cleanup_existing()
    col = get_or_create_collection()

    mats = build_materials()
    gull_mesh = build_gull_mesh(mats)
    arm_data = build_gull_armature_data(col)

    specs = _generate_flock_specs()
    _verify_and_adjust_paths(specs)

    rng = random.Random(42)

    for idx, spec in enumerate(specs):
        rig_name = f"{PREFIX}Rig_{idx:02d}"
        bird_name = f"{PREFIX}Bird_{idx:02d}"

        rig_obj = bpy.data.objects.new(name=rig_name, object_data=arm_data)
        col.objects.link(rig_obj)
        bpy.context.view_layer.update()

        bird_obj = bpy.data.objects.new(name=bird_name, object_data=gull_mesh)
        col.objects.link(bird_obj)
        bird_obj.parent = rig_obj

        _assign_vertex_weights(gull_mesh, bird_obj, populate_mesh=(idx == 0))

        arm_mod = bird_obj.modifiers.new(name="Armature", type="ARMATURE")
        arm_mod.object = rig_obj

        sub_mod = bird_obj.modifiers.new(name="Subdivision", type="SUBSURF")
        sub_mod.levels = 1
        sub_mod.render_levels = 2

        add_wing_flap_drivers(rig_obj, rng, spec["glide_bias"])
        animate_flight_path(rig_obj, spec)

        # Rename auto-created action with script prefix for clean idempotent cleanup
        if rig_obj.animation_data and rig_obj.animation_data.action:
            rig_obj.animation_data.action.name = f"{PREFIX}Action_{idx:02d}"

    return col


build()
