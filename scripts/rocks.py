"""Procedural coastal sea stacks and wave-washed boulders for Blender 5.3."""

import math
import random
import bpy
import bmesh
from mathutils import Vector, Euler


COLLECTION_NAME = "SeaStacks"
NAME_PREFIX = "SeaStacks_"
SEED = 7


def _cleanup_existing():
    """Remove existing SeaStacks collection and data-blocks created by this script."""
    col = bpy.data.collections.get(COLLECTION_NAME)
    if col is not None:
        objs = list(col.objects)
        for obj in objs:
            bpy.data.objects.remove(obj, do_unlink=True)
        for scene in bpy.data.scenes:
            if col.name in scene.collection.children:
                scene.collection.children.unlink(col)
        bpy.data.collections.remove(col)

    for mesh in list(bpy.data.meshes):
        if mesh.name.startswith(NAME_PREFIX) and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for mat in list(bpy.data.materials):
        if mat.name.startswith(NAME_PREFIX) and mat.users == 0:
            bpy.data.materials.remove(mat)

    for ng in list(bpy.data.node_groups):
        if ng.name.startswith(NAME_PREFIX) and ng.users == 0:
            bpy.data.node_groups.remove(ng)

    for act in list(bpy.data.actions):
        if act.name.startswith(NAME_PREFIX) and act.users == 0:
            bpy.data.actions.remove(act)


def _get_or_create_collection():
    """Create and link the dedicated SeaStacks collection."""
    col = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(col)
    return col


def _set_socket_default(node, socket_name, value):
    """Safely set a node input socket default value by name."""
    sock = node.inputs.get(socket_name)
    if sock is not None:
        sock.default_value = value


def _create_rock_geometry_nodes(name, is_stack=True):
    """Build a Geometry Nodes modifier group for layered cliffs or boulders."""
    ng = bpy.data.node_groups.new(name=name, type="GeometryNodeTree")

    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    in_node = nodes.new("NodeGroupInput")
    in_node.location = (-1200, 0)

    out_node = nodes.new("NodeGroupOutput")
    out_node.location = (900, 0)

    # Subdivision surface for smooth yet detailed displacement
    subsurf = nodes.new("GeometryNodeSubdivisionSurface")
    subsurf.location = (-950, 0)
    subsurf.inputs["Level"].default_value = 3 if is_stack else 2
    subsurf.inputs["Edge Crease"].default_value = 0.0
    links.new(in_node.outputs["Geometry"], subsurf.inputs["Mesh"])

    pos_node = nodes.new("GeometryNodeInputPosition")
    pos_node.location = (-1200, -350)

    norm_node = nodes.new("GeometryNodeInputNormal")
    norm_node.location = (-200, -200)

    # --- 1. Horizontal sedimentary strata (irregular bed thicknesses via 1D Noise on warped Z) ---
    warp_noise = nodes.new("ShaderNodeTexNoise")
    warp_noise.location = (-950, -300)
    warp_noise.inputs["Scale"].default_value = 0.04
    warp_noise.inputs["Detail"].default_value = 2.0
    links.new(pos_node.outputs["Position"], warp_noise.inputs["Vector"])

    sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
    sep_xyz.location = (-950, -520)
    links.new(pos_node.outputs["Position"], sep_xyz.inputs["Vector"])

    warp_scale = nodes.new("ShaderNodeMath")
    warp_scale.operation = "MULTIPLY"
    warp_scale.location = (-750, -350)
    warp_scale.inputs[1].default_value = 1.4 if is_stack else 0.4
    links.new(warp_noise.outputs["Fac"], warp_scale.inputs[0])

    warped_z = nodes.new("ShaderNodeMath")
    warped_z.operation = "ADD"
    warped_z.location = (-580, -450)
    links.new(sep_xyz.outputs["Z"], warped_z.inputs[0])
    links.new(warp_scale.outputs["Value"], warped_z.inputs[1])

    comb_strata = nodes.new("ShaderNodeCombineXYZ")
    comb_strata.location = (-400, -450)
    links.new(warped_z.outputs["Value"], comb_strata.inputs["Z"])

    # Coarse irregular sedimentary beds
    strata_noise_1d = nodes.new("ShaderNodeTexNoise")
    strata_noise_1d.location = (-220, -450)
    strata_noise_1d.noise_dimensions = "1D"
    strata_noise_1d.inputs["Scale"].default_value = 0.45 if is_stack else 0.30
    strata_noise_1d.inputs["Detail"].default_value = 4.0
    strata_noise_1d.inputs["Roughness"].default_value = 0.55
    links.new(warped_z.outputs["Value"], strata_noise_1d.inputs["W"])

    # Quantize slightly for flat-topped blocky ledges instead of sawtooth spikes
    strata_ramp = nodes.new("ShaderNodeValToRGB")
    strata_ramp.location = (-20, -450)
    strata_ramp.color_ramp.elements[0].position = 0.32
    strata_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    strata_ramp.color_ramp.elements[1].position = 0.68
    strata_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(strata_noise_1d.outputs["Fac"], strata_ramp.inputs["Fac"])

    strata_map = nodes.new("ShaderNodeMapRange")
    strata_map.location = (180, -450)
    strata_map.inputs["From Min"].default_value = 0.0
    strata_map.inputs["From Max"].default_value = 1.0
    strata_map.inputs["To Min"].default_value = -0.45 if is_stack else -0.12
    strata_map.inputs["To Max"].default_value = 0.55 if is_stack else 0.12
    links.new(strata_ramp.outputs["Color"], strata_map.inputs["Value"])

    # --- 2. Vertical cracks (Z-stretched Voronoi Distance to Edge) ---
    crack_scale_vec = nodes.new("ShaderNodeVectorMath")
    crack_scale_vec.operation = "MULTIPLY"
    crack_scale_vec.location = (-950, -760)
    crack_scale_vec.inputs[1].default_value = (0.24, 0.24, 0.035 if is_stack else 0.16)
    links.new(pos_node.outputs["Position"], crack_scale_vec.inputs[0])

    voronoi_cracks = nodes.new("ShaderNodeTexVoronoi")
    voronoi_cracks.location = (-720, -760)
    voronoi_cracks.feature = "DISTANCE_TO_EDGE"
    voronoi_cracks.inputs["Randomness"].default_value = 0.82
    links.new(crack_scale_vec.outputs["Vector"], voronoi_cracks.inputs["Vector"])

    crack_map = nodes.new("ShaderNodeMapRange")
    crack_map.location = (-500, -760)
    crack_map.clamp = True
    crack_map.inputs["From Min"].default_value = 0.0
    crack_map.inputs["From Max"].default_value = 0.065
    crack_map.inputs["To Min"].default_value = -0.70 if is_stack else -0.25
    crack_map.inputs["To Max"].default_value = 0.0
    links.new(voronoi_cracks.outputs["Distance"], crack_map.inputs["Value"])

    # --- 3. Medium/fine rocky crag breakup ---
    crag_noise = nodes.new("ShaderNodeTexNoise")
    crag_noise.location = (-720, -1020)
    crag_noise.inputs["Scale"].default_value = 0.15 if is_stack else 0.35
    crag_noise.inputs["Detail"].default_value = 4.0
    crag_noise.inputs["Roughness"].default_value = 0.55
    links.new(pos_node.outputs["Position"], crag_noise.inputs["Vector"])

    crag_map = nodes.new("ShaderNodeMapRange")
    crag_map.location = (-500, -1020)
    crag_map.inputs["From Min"].default_value = 0.22
    crag_map.inputs["From Max"].default_value = 0.78
    crag_map.inputs["To Min"].default_value = -0.42 if is_stack else -0.26
    crag_map.inputs["To Max"].default_value = 0.42 if is_stack else 0.26
    links.new(crag_noise.outputs["Fac"], crag_map.inputs["Value"])

    # Combine displacements
    add_1 = nodes.new("ShaderNodeMath")
    add_1.operation = "ADD"
    add_1.location = (160, -580)
    links.new(strata_map.outputs["Result"], add_1.inputs[0])
    links.new(crack_map.outputs["Result"], add_1.inputs[1])

    add_2 = nodes.new("ShaderNodeMath")
    add_2.operation = "ADD"
    add_2.location = (330, -680)
    links.new(add_1.outputs["Value"], add_2.inputs[0])
    links.new(crag_map.outputs["Result"], add_2.inputs[1])

    disp_vec = nodes.new("ShaderNodeVectorMath")
    disp_vec.operation = "SCALE"
    disp_vec.location = (480, -300)
    links.new(norm_node.outputs["Normal"], disp_vec.inputs[0])
    links.new(add_2.outputs["Value"], disp_vec.inputs["Scale"])

    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (520, 0)
    links.new(subsurf.outputs["Mesh"], set_pos.inputs["Geometry"])
    links.new(disp_vec.outputs["Vector"], set_pos.inputs["Offset"])

    shade_smooth = nodes.new("GeometryNodeSetShadeSmooth")
    shade_smooth.location = (710, 0)
    shade_smooth.inputs["Shade Smooth"].default_value = True
    links.new(set_pos.outputs["Geometry"], shade_smooth.inputs["Geometry"])
    links.new(shade_smooth.outputs["Geometry"], out_node.inputs["Geometry"])

    return ng


def _create_rock_material():
    """Build procedural sandstone/basalt material with strata, wetness, algae, barnacles, and guano."""
    mat = bpy.data.materials.new(name=f"{NAME_PREFIX}RockMat")
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1050, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Coordinates
    geom = nodes.new("ShaderNodeNewGeometry")
    geom.location = (-1600, 200)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1600, -200)

    sep_world = nodes.new("ShaderNodeSeparateXYZ")
    sep_world.location = (-1350, 350)
    links.new(geom.outputs["Position"], sep_world.inputs["Vector"])

    sep_norm = nodes.new("ShaderNodeSeparateXYZ")
    sep_norm.location = (-1350, 120)
    links.new(geom.outputs["Normal"], sep_norm.inputs["Vector"])

    sep_gen = nodes.new("ShaderNodeSeparateXYZ")
    sep_gen.location = (-1350, -200)
    links.new(tex_coord.outputs["Generated"], sep_gen.inputs["Vector"])

    # --- 1. Strata & Sandstone/Basalt Base Color ---
    strata_warp = nodes.new("ShaderNodeTexNoise")
    strata_warp.location = (-1350, 580)
    strata_warp.inputs["Scale"].default_value = 0.08
    strata_warp.inputs["Detail"].default_value = 3.0
    links.new(geom.outputs["Position"], strata_warp.inputs["Vector"])

    strata_z_add = nodes.new("ShaderNodeMath")
    strata_z_add.operation = "MULTIPLY_ADD"
    strata_z_add.location = (-1120, 480)
    strata_z_add.inputs[1].default_value = 2.2
    links.new(strata_warp.outputs["Fac"], strata_z_add.inputs[0])
    links.new(sep_world.outputs["Z"], strata_z_add.inputs[2])

    strata_comb = nodes.new("ShaderNodeCombineXYZ")
    strata_comb.location = (-930, 480)
    links.new(strata_z_add.outputs["Value"], strata_comb.inputs["Z"])

    strata_noise = nodes.new("ShaderNodeTexNoise")
    strata_noise.location = (-740, 480)
    strata_noise.inputs["Scale"].default_value = 0.65
    strata_noise.inputs["Detail"].default_value = 6.0
    strata_noise.inputs["Roughness"].default_value = 0.65
    links.new(strata_comb.outputs["Vector"], strata_noise.inputs["Vector"])

    sandstone_ramp = nodes.new("ShaderNodeValToRGB")
    sandstone_ramp.location = (-480, 520)
    cr = sandstone_ramp.color_ramp
    cr.elements[0].position = 0.18
    cr.elements[0].color = (0.14, 0.10, 0.08, 1.0)
    cr.elements[1].position = 0.82
    cr.elements[1].color = (0.38, 0.25, 0.16, 1.0)
    e1 = cr.elements.new(0.45)
    e1.color = (0.26, 0.18, 0.12, 1.0)
    e2 = cr.elements.new(0.62)
    e2.color = (0.20, 0.17, 0.15, 1.0)
    links.new(strata_noise.outputs["Fac"], sandstone_ramp.inputs["Fac"])

    # Basalt mix mask
    basalt_noise = nodes.new("ShaderNodeTexNoise")
    basalt_noise.location = (-740, 740)
    basalt_noise.inputs["Scale"].default_value = 0.14
    basalt_noise.inputs["Detail"].default_value = 5.0
    basalt_noise.inputs["Roughness"].default_value = 0.58
    links.new(geom.outputs["Position"], basalt_noise.inputs["Vector"])

    basalt_ramp = nodes.new("ShaderNodeValToRGB")
    basalt_ramp.location = (-480, 740)
    basalt_ramp.color_ramp.elements[0].position = 0.42
    basalt_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    basalt_ramp.color_ramp.elements[1].position = 0.64
    basalt_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(basalt_noise.outputs["Fac"], basalt_ramp.inputs["Fac"])

    mix_rock = nodes.new("ShaderNodeMix")
    mix_rock.data_type = "RGBA"
    mix_rock.location = (-180, 600)
    mix_rock.inputs["B"].default_value = (0.045, 0.048, 0.055, 1.0)
    links.new(basalt_ramp.outputs["Color"], mix_rock.inputs["Factor"])
    links.new(sandstone_ramp.outputs["Color"], mix_rock.inputs["A"])

    # --- 2. Bird-guano streaks on upper ledges ---
    guano_norm_map = nodes.new("ShaderNodeMapRange")
    guano_norm_map.location = (-740, 150)
    guano_norm_map.clamp = True
    guano_norm_map.inputs["From Min"].default_value = 0.25
    guano_norm_map.inputs["From Max"].default_value = 0.65
    links.new(sep_norm.outputs["Z"], guano_norm_map.inputs["Value"])

    guano_gen_map = nodes.new("ShaderNodeMapRange")
    guano_gen_map.location = (-740, -60)
    guano_gen_map.clamp = True
    guano_gen_map.inputs["From Min"].default_value = 0.45
    guano_gen_map.inputs["From Max"].default_value = 0.78
    links.new(sep_gen.outputs["Z"], guano_gen_map.inputs["Value"])

    guano_world_map = nodes.new("ShaderNodeMapRange")
    guano_world_map.location = (-740, -270)
    guano_world_map.clamp = True
    guano_world_map.inputs["From Min"].default_value = 5.5
    guano_world_map.inputs["From Max"].default_value = 10.0
    links.new(sep_world.outputs["Z"], guano_world_map.inputs["Value"])

    streak_coord = nodes.new("ShaderNodeVectorMath")
    streak_coord.operation = "MULTIPLY"
    streak_coord.location = (-930, 260)
    streak_coord.inputs[1].default_value = (1.6, 1.6, 0.08)
    links.new(geom.outputs["Position"], streak_coord.inputs[0])

    streak_noise = nodes.new("ShaderNodeTexNoise")
    streak_noise.location = (-740, 320)
    streak_noise.inputs["Scale"].default_value = 0.75
    streak_noise.inputs["Detail"].default_value = 5.0
    streak_noise.inputs["Roughness"].default_value = 0.65
    links.new(streak_coord.outputs["Vector"], streak_noise.inputs["Vector"])

    streak_ramp = nodes.new("ShaderNodeValToRGB")
    streak_ramp.location = (-520, 320)
    streak_ramp.color_ramp.elements[0].position = 0.42
    streak_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    streak_ramp.color_ramp.elements[1].position = 0.62
    streak_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(streak_noise.outputs["Fac"], streak_ramp.inputs["Fac"])

    guano_mul1 = nodes.new("ShaderNodeMath")
    guano_mul1.operation = "MULTIPLY"
    guano_mul1.location = (-480, 100)
    links.new(guano_norm_map.outputs["Result"], guano_mul1.inputs[0])
    links.new(guano_gen_map.outputs["Result"], guano_mul1.inputs[1])

    guano_mul2 = nodes.new("ShaderNodeMath")
    guano_mul2.operation = "MULTIPLY"
    guano_mul2.location = (-320, 20)
    links.new(guano_mul1.outputs["Value"], guano_mul2.inputs[0])
    links.new(guano_world_map.outputs["Result"], guano_mul2.inputs[1])

    guano_mask = nodes.new("ShaderNodeMath")
    guano_mask.operation = "MULTIPLY"
    guano_mask.location = (-160, 160)
    links.new(guano_mul2.outputs["Value"], guano_mask.inputs[0])
    links.new(streak_ramp.outputs["Color"], guano_mask.inputs[1])

    mix_guano = nodes.new("ShaderNodeMix")
    mix_guano.data_type = "RGBA"
    mix_guano.location = (60, 480)
    mix_guano.inputs["B"].default_value = (0.68, 0.65, 0.58, 1.0)
    links.new(guano_mask.outputs["Value"], mix_guano.inputs["Factor"])
    links.new(mix_rock.outputs["Result"], mix_guano.inputs["A"])

    # --- 3. Waterline algae & barnacle band (z in -1..1.2) ---
    splash_noise = nodes.new("ShaderNodeTexNoise")
    splash_noise.location = (-1120, -480)
    splash_noise.inputs["Scale"].default_value = 0.35
    splash_noise.inputs["Detail"].default_value = 4.0
    links.new(geom.outputs["Position"], splash_noise.inputs["Vector"])

    z_noisy = nodes.new("ShaderNodeMath")
    z_noisy.operation = "MULTIPLY_ADD"
    z_noisy.location = (-930, -480)
    z_noisy.inputs[1].default_value = 0.55
    links.new(splash_noise.outputs["Fac"], z_noisy.inputs[0])
    links.new(sep_world.outputs["Z"], z_noisy.inputs[2])

    band_low = nodes.new("ShaderNodeMapRange")
    band_low.location = (-740, -480)
    band_low.clamp = True
    band_low.inputs["From Min"].default_value = -1.2
    band_low.inputs["From Max"].default_value = -0.3
    links.new(z_noisy.outputs["Value"], band_low.inputs["Value"])

    band_high = nodes.new("ShaderNodeMapRange")
    band_high.location = (-740, -680)
    band_high.clamp = True
    band_high.inputs["From Min"].default_value = 0.7
    band_high.inputs["From Max"].default_value = 1.45
    band_high.inputs["To Min"].default_value = 1.0
    band_high.inputs["To Max"].default_value = 0.0
    links.new(z_noisy.outputs["Value"], band_high.inputs["Value"])

    waterline_band = nodes.new("ShaderNodeMath")
    waterline_band.operation = "MULTIPLY"
    waterline_band.location = (-520, -580)
    links.new(band_low.outputs["Result"], waterline_band.inputs[0])
    links.new(band_high.outputs["Result"], waterline_band.inputs[1])

    # Algae patches inside band
    algae_noise = nodes.new("ShaderNodeTexNoise")
    algae_noise.location = (-520, -780)
    algae_noise.inputs["Scale"].default_value = 1.6
    algae_noise.inputs["Detail"].default_value = 5.0
    links.new(geom.outputs["Position"], algae_noise.inputs["Vector"])

    algae_mask = nodes.new("ShaderNodeMath")
    algae_mask.operation = "MULTIPLY"
    algae_mask.location = (-300, -640)
    links.new(waterline_band.outputs["Value"], algae_mask.inputs[0])
    links.new(algae_noise.outputs["Fac"], algae_mask.inputs[1])

    mix_algae = nodes.new("ShaderNodeMix")
    mix_algae.data_type = "RGBA"
    mix_algae.location = (280, 420)
    mix_algae.inputs["B"].default_value = (0.025, 0.065, 0.022, 1.0)
    links.new(algae_mask.outputs["Value"], mix_algae.inputs["Factor"])
    links.new(mix_guano.outputs["Result"], mix_algae.inputs["A"])

    # Barnacle speckles inside band
    barnacle_vor = nodes.new("ShaderNodeTexVoronoi")
    barnacle_vor.location = (-520, -1000)
    barnacle_vor.inputs["Scale"].default_value = 22.0
    links.new(geom.outputs["Position"], barnacle_vor.inputs["Vector"])

    barnacle_ramp = nodes.new("ShaderNodeValToRGB")
    barnacle_ramp.location = (-300, -1000)
    barnacle_ramp.color_ramp.elements[0].position = 0.22
    barnacle_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    barnacle_ramp.color_ramp.elements[1].position = 0.34
    barnacle_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(barnacle_vor.outputs["Distance"], barnacle_ramp.inputs["Fac"])

    barnacle_mask = nodes.new("ShaderNodeMath")
    barnacle_mask.operation = "MULTIPLY"
    barnacle_mask.location = (-60, -860)
    links.new(waterline_band.outputs["Value"], barnacle_mask.inputs[0])
    links.new(barnacle_ramp.outputs["Color"], barnacle_mask.inputs[1])

    mix_barnacles = nodes.new("ShaderNodeMix")
    mix_barnacles.data_type = "RGBA"
    mix_barnacles.location = (480, 360)
    mix_barnacles.inputs["B"].default_value = (0.36, 0.35, 0.31, 1.0)
    links.new(barnacle_mask.outputs["Value"], mix_barnacles.inputs["Factor"])
    links.new(mix_algae.outputs["Result"], mix_barnacles.inputs["A"])

    # --- 4. Wet darkening & glossiness below z=2 (strongest z < 1.5) ---
    wet_map = nodes.new("ShaderNodeMapRange")
    wet_map.location = (-520, -340)
    wet_map.clamp = True
    wet_map.inputs["From Min"].default_value = 1.15
    wet_map.inputs["From Max"].default_value = 2.35
    wet_map.inputs["To Min"].default_value = 1.0
    wet_map.inputs["To Max"].default_value = 0.0
    links.new(z_noisy.outputs["Value"], wet_map.inputs["Value"])

    wet_darken = nodes.new("ShaderNodeHueSaturation")
    wet_darken.location = (680, 360)
    wet_darken.inputs["Value"].default_value = 0.24
    wet_darken.inputs["Saturation"].default_value = 1.18
    links.new(wet_map.outputs["Result"], wet_darken.inputs["Fac"])
    links.new(mix_barnacles.outputs["Result"], wet_darken.inputs["Color"])
    links.new(wet_darken.outputs["Color"], bsdf.inputs["Base Color"])

    # --- 5. Roughness variation ---
    rough_noise = nodes.new("ShaderNodeTexNoise")
    rough_noise.location = (100, -180)
    rough_noise.inputs["Scale"].default_value = 4.5
    rough_noise.inputs["Detail"].default_value = 6.0
    links.new(geom.outputs["Position"], rough_noise.inputs["Vector"])

    dry_rough = nodes.new("ShaderNodeMapRange")
    dry_rough.location = (320, -180)
    dry_rough.inputs["From Min"].default_value = 0.25
    dry_rough.inputs["From Max"].default_value = 0.75
    dry_rough.inputs["To Min"].default_value = 0.62
    dry_rough.inputs["To Max"].default_value = 0.92
    links.new(rough_noise.outputs["Fac"], dry_rough.inputs["Value"])

    mix_rough = nodes.new("ShaderNodeMix")
    mix_rough.data_type = "FLOAT"
    mix_rough.location = (680, -80)
    mix_rough.inputs["B"].default_value = 0.08
    links.new(wet_map.outputs["Result"], mix_rough.inputs["Factor"])
    links.new(dry_rough.outputs["Result"], mix_rough.inputs["A"])
    links.new(mix_rough.outputs["Result"], bsdf.inputs["Roughness"])

    # Subtle wet clearcoat
    coat_mul = nodes.new("ShaderNodeMath")
    coat_mul.operation = "MULTIPLY"
    coat_mul.location = (680, -260)
    coat_mul.inputs[1].default_value = 0.55
    links.new(wet_map.outputs["Result"], coat_mul.inputs[0])
    links.new(coat_mul.outputs["Value"], bsdf.inputs["Coat Weight"])

    # --- 6. Multi-scale Bump ---
    micro_noise = nodes.new("ShaderNodeTexNoise")
    micro_noise.location = (100, -450)
    micro_noise.inputs["Scale"].default_value = 7.5
    micro_noise.inputs["Detail"].default_value = 10.0
    micro_noise.inputs["Roughness"].default_value = 0.72
    links.new(geom.outputs["Position"], micro_noise.inputs["Vector"])

    fine_noise = nodes.new("ShaderNodeTexNoise")
    fine_noise.location = (100, -680)
    fine_noise.inputs["Scale"].default_value = 28.0
    fine_noise.inputs["Detail"].default_value = 6.0
    fine_noise.inputs["Roughness"].default_value = 0.65
    links.new(geom.outputs["Position"], fine_noise.inputs["Vector"])

    bump_add0 = nodes.new("ShaderNodeMath")
    bump_add0.operation = "MULTIPLY_ADD"
    bump_add0.location = (300, -620)
    bump_add0.inputs[1].default_value = 0.45
    links.new(fine_noise.outputs["Fac"], bump_add0.inputs[0])
    links.new(micro_noise.outputs["Fac"], bump_add0.inputs[2])

    bump_add1 = nodes.new("ShaderNodeMath")
    bump_add1.operation = "MULTIPLY_ADD"
    bump_add1.location = (470, -480)
    bump_add1.inputs[1].default_value = 0.40
    links.new(strata_noise.outputs["Fac"], bump_add1.inputs[0])
    links.new(bump_add0.outputs["Value"], bump_add1.inputs[2])

    bump_add2 = nodes.new("ShaderNodeMath")
    bump_add2.operation = "MULTIPLY_ADD"
    bump_add2.location = (640, -480)
    bump_add2.inputs[1].default_value = 0.55
    links.new(barnacle_mask.outputs["Value"], bump_add2.inputs[0])
    links.new(bump_add1.outputs["Value"], bump_add2.inputs[2])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (820, -450)
    bump.inputs["Strength"].default_value = 0.85
    bump.inputs["Distance"].default_value = 0.16
    links.new(bump_add2.outputs["Value"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _build_stack_bmesh(rng, base_radius, height):
    """Create a layered, undercut sea stack base mesh extending down to z=-8."""
    bm = bmesh.new()
    seg_count = 24
    z_bottom = -8.0

    # Vertical profile stations (z, radius_multiplier)
    z_levels = [
        z_bottom,
        -4.5,
        -1.5,
        0.2,
        1.4,
        2.6,
        3.8,
        5.5,
        height * 0.25,
        height * 0.36,
        height * 0.47,
        height * 0.58,
        height * 0.69,
        height * 0.80,
        height * 0.90,
        height,
    ]
    z_levels = sorted(set(round(z, 3) for z in z_levels))

    # Angular harmonic phases for faceted cliff buttresses
    p2 = rng.uniform(0, math.tau)
    p3 = rng.uniform(0, math.tau)
    p5 = rng.uniform(0, math.tau)
    lean_x = rng.uniform(-0.035, 0.035)
    lean_y = rng.uniform(-0.035, 0.035)

    rings = []
    for idx, z in enumerate(z_levels):
        # Base taper + pronounced wave-cut erosion notch near z = 0.5..2.4
        if z < -1.0:
            profile = 1.18 - 0.04 * (z + 8.0) / 7.0
        elif z < 3.6:
            notch = math.exp(-((z - 1.35) / 1.15) ** 2)
            profile = 1.06 - 0.34 * notch
        else:
            step_jitter = rng.uniform(-0.06, 0.07)
            profile = (1.06 - 0.60 * ((z - 3.6) / max(1.0, height - 3.6)) ** 1.15) + step_jitter

        ring_verts = []
        cx = lean_x * max(0.0, z)
        cy = lean_y * max(0.0, z)
        for s in range(seg_count):
            ang = (s / seg_count) * math.tau
            ang_mod = (
                1.0
                + 0.16 * math.cos(2.0 * ang + p2)
                + 0.11 * math.cos(3.0 * ang + p3)
                + 0.05 * math.cos(5.0 * ang + p5)
                + rng.uniform(-0.04, 0.04)
            )
            r = max(0.8, base_radius * profile * ang_mod)
            vx = cx + r * math.cos(ang)
            vy = cy + r * math.sin(ang)
            vz = z + (rng.uniform(-0.18, 0.18) if 0 < idx < len(z_levels) - 1 else 0.0)
            ring_verts.append(bm.verts.new((vx, vy, vz)))
        rings.append(ring_verts)

    # Bridge side faces
    for r_i in range(len(rings) - 1):
        r0 = rings[r_i]
        r1 = rings[r_i + 1]
        for s in range(seg_count):
            s_next = (s + 1) % seg_count
            bm.faces.new((r0[s], r0[s_next], r1[s_next], r1[s]))

    # Cap bottom
    bm.faces.new(list(reversed(rings[0])))

    # Cap top with craggy summit peak
    top_ring = rings[-1]
    top_center = sum((v.co for v in top_ring), Vector()) / len(top_ring)
    top_center.z += rng.uniform(0.6, 1.8)
    apex = bm.verts.new(top_center)
    for s in range(seg_count):
        s_next = (s + 1) % seg_count
        bm.faces.new((top_ring[s], top_ring[s_next], apex))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _build_boulder_bmesh(rng, radius, top_z):
    """Create a wave-washed reef/boulder base mesh extending down to z=-8."""
    bm = bmesh.new()
    seg_count = 12
    z_bottom = -8.0

    z_levels = [
        z_bottom,
        -4.0,
        -1.2,
        0.0,
        top_z * 0.45,
        top_z * 0.82,
        top_z,
    ]
    z_levels = sorted(set(round(z, 3) for z in z_levels))

    p2 = rng.uniform(0, math.tau)
    p3 = rng.uniform(0, math.tau)

    rings = []
    for idx, z in enumerate(z_levels):
        if z <= 0.0:
            profile = 1.05 + 0.15 * (-z / 8.0)
        else:
            t = z / max(0.4, top_z)
            profile = max(0.28, math.sqrt(max(0.05, 1.0 - 0.72 * t * t)))

        ring_verts = []
        for s in range(seg_count):
            ang = (s / seg_count) * math.tau
            ang_mod = (
                1.0
                + 0.18 * math.cos(2.0 * ang + p2)
                + 0.10 * math.cos(3.0 * ang + p3)
                + rng.uniform(-0.06, 0.06)
            )
            r = max(0.5, radius * profile * ang_mod)
            vx = r * math.cos(ang)
            vy = r * math.sin(ang)
            ring_verts.append(bm.verts.new((vx, vy, z)))
        rings.append(ring_verts)

    for r_i in range(len(rings) - 1):
        r0 = rings[r_i]
        r1 = rings[r_i + 1]
        for s in range(seg_count):
            s_next = (s + 1) % seg_count
            bm.faces.new((r0[s], r0[s_next], r1[s_next], r1[s]))

    bm.faces.new(list(reversed(rings[0])))
    top_ring = rings[-1]
    top_center = sum((v.co for v in top_ring), Vector()) / len(top_ring)
    top_center.z += rng.uniform(0.15, 0.45)
    apex = bm.verts.new(top_center)
    for s in range(seg_count):
        s_next = (s + 1) % seg_count
        bm.faces.new((top_ring[s], top_ring[s_next], apex))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _is_valid_placement(x, y, radius, placed):
    """Check bounds, lighthouse clearance (25 m), camera corridor, and overlap."""
    if not (-90.0 + radius <= x <= 90.0 - radius):
        return False
    if not (40.0 + radius <= y <= 170.0 - radius):
        return False

    # Keep 25 m radius clear around lighthouse islet at (10, 80)
    if math.hypot(x - 10.0, y - 80.0) < 25.0 + radius + 1.5:
        return False

    # Keep camera corridor x in [-35, 0], y in [-90, 40] clear
    if -35.0 - radius <= x <= 0.0 + radius and y < 40.0 + radius + 2.0:
        return False

    # Avoid awkward overlapping with already placed rocks
    for px, py, pr in placed:
        if math.hypot(x - px, y - py) < radius + pr + 2.5:
            return False

    return True


def _create_rock_object(name, bm, location, rot_z, col, mat, geo_node_group):
    """Create mesh object, link to collection, assign material and GeoNodes."""
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = Euler((0.0, 0.0, rot_z), "XYZ")
    col.objects.link(obj)

    obj.data.materials.append(mat)

    mod = obj.modifiers.new(name="RockDetail", type="NODES")
    mod.node_group = geo_node_group
    return obj


def build():
    """Idempotently build the SeaStacks collection with sea stacks and boulders."""
    _cleanup_existing()
    col = _get_or_create_collection()
    rng = random.Random(SEED)

    mat = _create_rock_material()
    stack_gn = _create_rock_geometry_nodes(f"{NAME_PREFIX}StackGeo", is_stack=True)
    boulder_gn = _create_rock_geometry_nodes(f"{NAME_PREFIX}BoulderGeo", is_stack=False)

    placed = []

    # 7 tall sea stacks (8-35 m high), including two hero stacks at (-45, 110) and (55, 140)
    stack_specs = [
        (-45.0, 110.0, 9.5, 32.0),   # Hero West stack
        (55.0, 140.0, 10.5, 34.0),   # Hero East stack
        (-68.0, 72.0, 7.2, 22.0),    # Mid-West sentinel
        (-76.0, 150.0, 8.2, 26.5),   # Far-West background
        (64.0, 66.0, 6.8, 18.5),     # Mid-East stack
        (-14.0, 156.0, 7.8, 24.5),   # North-Central distant stack
        (36.0, 116.0, 5.4, 12.5),    # Smaller eastern spire
    ]

    for i, (x, y, radius, height) in enumerate(stack_specs):
        bm = _build_stack_bmesh(rng, radius, height)
        _create_rock_object(
            name=f"{NAME_PREFIX}Stack_{i + 1:02d}",
            bm=bm,
            location=(x, y, 0.0),
            rot_z=rng.uniform(0.0, math.tau),
            col=col,
            mat=mat,
            geo_node_group=stack_gn,
        )
        placed.append((x, y, radius))

    # 13 smaller wave-washed boulders/reefs near the surface
    boulder_target = 13
    attempts = 0
    boulder_idx = 0
    while boulder_idx < boulder_target and attempts < 500:
        attempts += 1
        radius = rng.uniform(2.0, 5.0)
        x = rng.uniform(-84.0, 84.0)
        y = rng.uniform(44.0, 165.0)
        if not _is_valid_placement(x, y, radius, placed):
            continue

        top_z = rng.uniform(0.9, 4.0)
        bm = _build_boulder_bmesh(rng, radius, top_z)
        _create_rock_object(
            name=f"{NAME_PREFIX}Boulder_{boulder_idx + 1:02d}",
            bm=bm,
            location=(x, y, 0.0),
            rot_z=rng.uniform(0.0, math.tau),
            col=col,
            mat=mat,
            geo_node_group=boulder_gn,
        )
        placed.append((x, y, radius))
        boulder_idx += 1

    return col


build()
