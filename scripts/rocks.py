"""Procedural coastal sea stacks and angular reef boulders for Blender 5.3."""

import math
import random
import bpy
import bmesh
from mathutils import Vector, Euler, Matrix


COLLECTION_NAME = "SeaStacks"
NAME_PREFIX = "SeaStacks_"
SEED = 19


def _cleanup_existing():
    """Remove existing SeaStacks collection and data-blocks created by this script."""
    col = bpy.data.collections.get(COLLECTION_NAME)
    if col is not None:
        for obj in list(col.objects):
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


def _bisect_and_cap(bm, plane_co, plane_no):
    """Slice a closed bmesh with a plane (clearing outer) and cap the cut loop."""
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    ret = bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        dist=1e-4,
        plane_co=plane_co,
        plane_no=plane_no,
        clear_outer=True,
        clear_inner=False,
    )
    cut_edges = [e for e in ret.get("geom_cut", []) if isinstance(e, bmesh.types.BMEdge)]
    if cut_edges:
        bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)


def _append_bmesh(dst_bm, src_bm):
    """Copy all vertices and faces from src_bm into dst_bm."""
    vert_map = {}
    for v in src_bm.verts:
        vert_map[v.index] = dst_bm.verts.new(v.co)
    for f in src_bm.faces:
        verts = [vert_map[v.index] for v in f.verts]
        try:
            dst_bm.faces.new(verts)
        except ValueError:
            pass


def _build_fractured_column_part(
    rng,
    offset_xy,
    rx,
    ry,
    z_min,
    z_max,
    rot_z,
    lean_xy,
    top_tilt_deg,
    undercut_dir,
    overhang_bias=0.0,
):
    """Build one closed, tilted, plane-bisected irregular rock column/buttress."""
    bm = bmesh.new()
    sides = rng.randint(7, 10)
    z_stations = [
        z_min,
        -2.0,
        -0.4,
        1.1,
        2.5,
        z_max * 0.35,
        z_max * 0.58,
        z_max * 0.80,
        z_max + 2.5,
    ]
    z_stations = sorted(set(round(z, 3) for z in z_stations if z >= z_min))

    # Irregular non-circular cross-section angles & radii
    raw_angles = sorted(rng.uniform(0.0, math.tau) for _ in range(sides))
    # Enforce minimum angular spacing so faces don't collapse
    angles = []
    for i, a in enumerate(raw_angles):
        min_a = (i / sides) * math.tau + 0.12
        max_a = ((i + 1) / sides) * math.tau - 0.12
        angles.append(max(min_a, min(max_a, a)))

    radial_jitter = [rng.uniform(0.68, 1.32) for _ in range(sides)]
    height_scale_mod = [rng.uniform(0.84, 1.18) for _ in z_stations]

    rot_mat = Matrix.Rotation(rot_z, 3, "Z")
    ox, oy = offset_xy
    lx, ly = lean_xy

    rings = []
    for zi, z in enumerate(z_stations):
        t = max(0.0, (z - z_min) / max(1.0, z_max - z_min))
        # Non-monotonic taper: wider base skirt, optional mid-column bulge/overhang
        if z < 0.0:
            base_scale = 1.16 - 0.12 * ((z - z_min) / max(0.5, -z_min))
        else:
            t_above = min(1.2, z / max(1.0, z_max))
            base_scale = 1.0 - 0.28 * (t_above ** 1.25) + overhang_bias * math.sin(t_above * math.pi)
        base_scale *= height_scale_mod[zi]

        # Asymmetric wave-cut notch near z = -0.5..2.6 stronger on undercut_dir side
        notch_z_weight = math.exp(-((z - 0.85) / 1.15) ** 2) if -1.2 < z < 3.2 else 0.0

        ring = []
        for si, ang in enumerate(angles):
            dir_align = max(0.0, math.cos(ang - undercut_dir))
            notch_factor = 1.0 - notch_z_weight * (0.14 + 0.30 * (dir_align ** 1.6))

            local_x = rx * radial_jitter[si] * base_scale * notch_factor * math.cos(ang)
            local_y = ry * radial_jitter[si] * base_scale * notch_factor * math.sin(ang)

            p = rot_mat @ Vector((local_x, local_y, 0.0))
            p.x += ox + lx * max(0.0, z)
            p.y += oy + ly * max(0.0, z)
            p.z = z
            ring.append(bm.verts.new(p))
        rings.append(ring)

    for ri in range(len(rings) - 1):
        r0 = rings[ri]
        r1 = rings[ri + 1]
        for si in range(sides):
            sn = (si + 1) % sides
            bm.faces.new((r0[si], r0[sn], r1[sn], r1[si]))

    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    center_top = Vector((ox + lx * z_max, oy + ly * z_max, z_max))

    # 1. Fractured flat or sloped top cut (no symmetric cone)
    top_tilt_rad = math.radians(top_tilt_deg)
    top_az = rng.uniform(0.0, math.tau)
    top_no = Vector((
        math.sin(top_tilt_rad) * math.cos(top_az),
        math.sin(top_tilt_rad) * math.sin(top_az),
        math.cos(top_tilt_rad),
    )).normalized()
    _bisect_and_cap(bm, center_top, top_no)

    # 2. Sheer vertical / diagonal joint cuts (1-2 flat cliff walls or overhangs)
    num_sheer = rng.randint(1, 3)
    avg_r = 0.5 * (rx + ry)
    for _ in range(num_sheer):
        cut_az = rng.uniform(0.0, math.tau)
        # Negative z normal slices inward toward the base -> overhangs!
        cut_tilt_z = rng.uniform(-0.22, 0.24)
        cut_no = Vector((math.cos(cut_az), math.sin(cut_az), cut_tilt_z)).normalized()
        cut_z = rng.uniform(z_max * 0.25, z_max * 0.85)
        cut_dist = avg_r * rng.uniform(0.55, 0.92)
        cut_co = Vector((
            ox + lx * cut_z + math.cos(cut_az) * cut_dist,
            oy + ly * cut_z + math.sin(cut_az) * cut_dist,
            cut_z,
        ))
        _bisect_and_cap(bm, cut_co, cut_no)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _build_stack_bmesh(rng, base_radius, height, archetype):
    """Compose a sea stack from multiple asymmetric, tilted, fractured rock columns."""
    bm = bmesh.new()
    z_min = -6.5
    undercut_dir = rng.uniform(0.0, math.tau)
    main_rot = rng.uniform(0.0, math.tau)

    # Lean vector shared roughly with slight variation per buttress
    lean_mag = rng.uniform(0.02, 0.075)
    lean_ang = rng.uniform(0.0, math.tau)
    main_lean = (lean_mag * math.cos(lean_ang), lean_mag * math.sin(lean_ang))

    # 1. Main tower (elongated slab, never circular)
    aspect = rng.uniform(1.30, 1.85)
    main_rx = base_radius * 0.72 * math.sqrt(aspect)
    main_ry = base_radius * 0.72 / math.sqrt(aspect)
    main_col = _build_fractured_column_part(
        rng=rng,
        offset_xy=(0.0, 0.0),
        rx=main_rx,
        ry=main_ry,
        z_min=z_min,
        z_max=height,
        rot_z=main_rot,
        lean_xy=main_lean,
        top_tilt_deg=rng.uniform(4.0, 22.0),
        undercut_dir=undercut_dir,
        overhang_bias=rng.uniform(-0.08, 0.22),
    )
    _append_bmesh(bm, main_col)
    main_col.free()

    # 2. Primary shoulder / buttress (gives stepped asymmetric sea stack silhouette)
    sh_ang = main_rot + rng.uniform(0.4, 2.3) * (1.0 if archetype % 2 == 0 else -1.0)
    sh_dist = base_radius * rng.uniform(0.42, 0.68)
    sh_height = height * rng.uniform(0.46, 0.78)
    sh_aspect = rng.uniform(1.15, 1.75)
    sh_rx = base_radius * 0.56 * math.sqrt(sh_aspect)
    sh_ry = base_radius * 0.56 / math.sqrt(sh_aspect)
    sh_lean = (
        main_lean[0] + rng.uniform(-0.03, 0.03),
        main_lean[1] + rng.uniform(-0.03, 0.03),
    )
    shoulder = _build_fractured_column_part(
        rng=rng,
        offset_xy=(sh_dist * math.cos(sh_ang), sh_dist * math.sin(sh_ang)),
        rx=sh_rx,
        ry=sh_ry,
        z_min=z_min,
        z_max=sh_height,
        rot_z=main_rot + rng.uniform(-1.1, 1.1),
        lean_xy=sh_lean,
        top_tilt_deg=rng.uniform(6.0, 26.0),
        undercut_dir=undercut_dir + rng.uniform(-0.5, 0.5),
        overhang_bias=rng.uniform(0.0, 0.18),
    )
    _append_bmesh(bm, shoulder)
    shoulder.free()

    # 3. Secondary lower spur / canted crag on opposite flank
    sp_ang = sh_ang + rng.uniform(1.8, 3.6)
    sp_dist = base_radius * rng.uniform(0.40, 0.72)
    sp_height = height * rng.uniform(0.24, 0.52)
    spur = _build_fractured_column_part(
        rng=rng,
        offset_xy=(sp_dist * math.cos(sp_ang), sp_dist * math.sin(sp_ang)),
        rx=base_radius * rng.uniform(0.38, 0.58),
        ry=base_radius * rng.uniform(0.30, 0.48),
        z_min=z_min,
        z_max=sp_height,
        rot_z=rng.uniform(0.0, math.tau),
        lean_xy=(rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)),
        top_tilt_deg=rng.uniform(12.0, 34.0),
        undercut_dir=undercut_dir,
        overhang_bias=rng.uniform(-0.05, 0.14),
    )
    _append_bmesh(bm, spur)
    spur.free()

    # 4. Irregular submerged reef plinth (anchors stack naturally into waves)
    plinth = _build_fractured_column_part(
        rng=rng,
        offset_xy=(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)),
        rx=base_radius * rng.uniform(0.92, 1.12),
        ry=base_radius * rng.uniform(0.76, 0.96),
        z_min=z_min,
        z_max=rng.uniform(0.4, 1.5),
        rot_z=rng.uniform(0.0, math.tau),
        lean_xy=(0.0, 0.0),
        top_tilt_deg=rng.uniform(8.0, 22.0),
        undercut_dir=undercut_dir,
        overhang_bias=0.0,
    )
    _append_bmesh(bm, plinth)
    plinth.free()

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _build_boulder_bmesh(rng, radius, top_z):
    """Create an angular, plane-fractured reef rock chunk extending down to z=-4.5."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)

    sx = radius * rng.uniform(0.85, 1.35)
    sy = radius * rng.uniform(0.55, 0.95)
    z_bottom = -4.5
    sz = (top_z - z_bottom) * 0.58

    tilt_x = rng.uniform(-0.45, 0.45)
    tilt_y = rng.uniform(-0.45, 0.45)
    rot_z = rng.uniform(0.0, math.tau)
    mat_scale = Matrix.Diagonal((sx, sy, sz, 1.0))
    mat_rot = Euler((tilt_x, tilt_y, rot_z), "XYZ").to_matrix().to_4x4()
    mat_trans = Matrix.Translation((0.0, 0.0, (top_z + z_bottom) * 0.5))
    bmesh.ops.transform(bm, matrix=mat_trans @ mat_rot @ mat_scale, verts=bm.verts)

    # Slice flat fracture planes across the top and flanks for crisp angular reef teeth
    # Top oblique cleavage cuts near top_z
    num_top_cuts = rng.randint(2, 3)
    for _ in range(num_top_cuts):
        tilt_rad = math.radians(rng.uniform(10.0, 42.0))
        az = rng.uniform(0.0, math.tau)
        no = Vector((
            math.sin(tilt_rad) * math.cos(az),
            math.sin(tilt_rad) * math.sin(az),
            math.cos(tilt_rad),
        )).normalized()
        co = Vector((
            rng.uniform(-0.25, 0.25) * radius,
            rng.uniform(-0.25, 0.25) * radius,
            top_z * rng.uniform(0.85, 1.05),
        ))
        _bisect_and_cap(bm, co, no)

    # Steep flank cleavage planes
    num_side_cuts = rng.randint(5, 8)
    for k in range(num_side_cuts):
        az = (k / num_side_cuts) * math.tau + rng.uniform(-0.35, 0.35)
        dip = rng.uniform(-0.18, 0.48)
        no = Vector((math.cos(az), math.sin(az), dip)).normalized()
        dist = radius * rng.uniform(0.45, 0.88)
        co = Vector((math.cos(az) * dist, math.sin(az) * dist, rng.uniform(-0.6, top_z * 0.7)))
        _bisect_and_cap(bm, co, no)

    # Clamp bottom flat at z_bottom so underwater extent is bounded cleanly
    _bisect_and_cap(bm, Vector((0.0, 0.0, z_bottom)), Vector((0.0, 0.0, -1.0)))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _add_geonode_input(ng, name, socket_type, default_value=None):
    """Create a Geometry Nodes group input socket and return its socket descriptor."""
    sock = ng.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
    if default_value is not None and hasattr(sock, "default_value"):
        sock.default_value = default_value
    return sock


def _create_rock_geometry_nodes(name, is_stack=True):
    """Build Geometry Nodes displacement tree with multi-scale 3D noise and Voronoi fractures."""
    ng = bpy.data.node_groups.new(name=name, type="GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    seed_sock = _add_geonode_input(ng, "SeedOffset", "NodeSocketVector", (0.0, 0.0, 0.0))
    tilt_sock = _add_geonode_input(ng, "StrataTilt", "NodeSocketVector", (0.14, -0.08, 0.0))
    notch_sock = _add_geonode_input(ng, "NotchDepth", "NodeSocketFloat", 1.35 if is_stack else 0.0)

    nodes = ng.nodes
    links = ng.links

    in_node = nodes.new("NodeGroupInput")
    in_node.location = (-1800, 0)
    out_node = nodes.new("NodeGroupOutput")
    out_node.location = (1500, 0)

    # Pre-conditioning:
    # Stacks come from VOXEL Remesh -> light Blur on Position removes voxel stair-steps.
    # Boulders come from angular bisected polyhedra -> Simple SubdivideMesh preserves sharp ridges.
    if is_stack:
        pos_in = nodes.new("GeometryNodeInputPosition")
        pos_in.location = (-1600, 200)

        blur_pos = nodes.new("GeometryNodeBlurAttribute")
        blur_pos.data_type = "FLOAT_VECTOR"
        blur_pos.location = (-1400, 200)
        blur_pos.inputs["Iterations"].default_value = 2
        blur_pos.inputs["Weight"].default_value = 0.85
        links.new(pos_in.outputs["Position"], blur_pos.inputs["Value"])

        prep_geo = nodes.new("GeometryNodeSetPosition")
        prep_geo.location = (-1200, 0)
        links.new(in_node.outputs["Geometry"], prep_geo.inputs["Geometry"])
        links.new(blur_pos.outputs["Value"], prep_geo.inputs["Position"])
    else:
        prep_geo = nodes.new("GeometryNodeSubdivideMesh")
        prep_geo.location = (-1200, 0)
        prep_geo.inputs["Level"].default_value = 3
        links.new(in_node.outputs["Geometry"], prep_geo.inputs["Mesh"])

    pos_node = nodes.new("GeometryNodeInputPosition")
    pos_node.location = (-1550, -320)
    norm_node = nodes.new("GeometryNodeInputNormal")
    norm_node.location = (-1550, -140)

    # Seeded coordinate = Position + SeedOffset
    seeded_pos = nodes.new("ShaderNodeVectorMath")
    seeded_pos.operation = "ADD"
    seeded_pos.location = (-1320, -340)
    links.new(pos_node.outputs["Position"], seeded_pos.inputs[0])
    links.new(in_node.outputs[seed_sock.name], seeded_pos.inputs[1])

    # --- 1. Large Low-Frequency Asymmetric Shape Warp (3D, non-radial) ---
    macro_noise = nodes.new("ShaderNodeTexNoise")
    macro_noise.location = (-1080, -180)
    macro_noise.inputs["Scale"].default_value = 0.055 if is_stack else 0.22
    macro_noise.inputs["Detail"].default_value = 2.5
    macro_noise.inputs["Roughness"].default_value = 0.50
    links.new(seeded_pos.outputs["Vector"], macro_noise.inputs["Vector"])

    macro_center = nodes.new("ShaderNodeVectorMath")
    macro_center.operation = "SUBTRACT"
    macro_center.location = (-860, -180)
    macro_center.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(macro_noise.outputs["Color"], macro_center.inputs[0])

    macro_vec = nodes.new("ShaderNodeVectorMath")
    macro_vec.operation = "MULTIPLY"
    macro_vec.location = (-660, -180)
    macro_vec.inputs[1].default_value = (2.4, 2.4, 0.85) if is_stack else (0.14, 0.14, 0.08)
    links.new(macro_center.outputs["Vector"], macro_vec.inputs[0])

    # --- 2. Asymmetric Wave-Cut Notch near waterline (stacks only) ---
    sep_pos = nodes.new("ShaderNodeSeparateXYZ")
    sep_pos.location = (-1320, -560)
    links.new(pos_node.outputs["Position"], sep_pos.inputs["Vector"])

    # Gaussian bell around z = 0.75 m
    z_sub = nodes.new("ShaderNodeMath")
    z_sub.operation = "SUBTRACT"
    z_sub.location = (-1120, -560)
    z_sub.inputs[1].default_value = 0.75
    links.new(sep_pos.outputs["Z"], z_sub.inputs[0])

    z_div = nodes.new("ShaderNodeMath")
    z_div.operation = "DIVIDE"
    z_div.location = (-940, -560)
    z_div.inputs[1].default_value = 1.30
    links.new(z_sub.outputs["Value"], z_div.inputs[0])

    z_sq = nodes.new("ShaderNodeMath")
    z_sq.operation = "POWER"
    z_sq.location = (-760, -560)
    z_sq.inputs[1].default_value = 2.0
    links.new(z_div.outputs["Value"], z_sq.inputs[0])

    z_neg = nodes.new("ShaderNodeMath")
    z_neg.operation = "MULTIPLY"
    z_neg.location = (-580, -560)
    z_neg.inputs[1].default_value = -1.0
    links.new(z_sq.outputs["Value"], z_neg.inputs[0])

    z_bell = nodes.new("ShaderNodeMath")
    z_bell.operation = "EXPONENT"
    z_bell.location = (-400, -560)
    links.new(z_neg.outputs["Value"], z_bell.inputs[0])

    notch_noise = nodes.new("ShaderNodeTexNoise")
    notch_noise.location = (-1080, -760)
    notch_noise.inputs["Scale"].default_value = 0.11
    notch_noise.inputs["Detail"].default_value = 3.0
    links.new(seeded_pos.outputs["Vector"], notch_noise.inputs["Vector"])

    notch_mask_map = nodes.new("ShaderNodeMapRange")
    notch_mask_map.location = (-860, -760)
    notch_mask_map.clamp = True
    notch_mask_map.inputs["From Min"].default_value = 0.30
    notch_mask_map.inputs["From Max"].default_value = 0.72
    notch_mask_map.inputs["To Min"].default_value = 0.12
    notch_mask_map.inputs["To Max"].default_value = 1.0
    links.new(notch_noise.outputs["Fac"], notch_mask_map.inputs["Value"])

    notch_mul1 = nodes.new("ShaderNodeMath")
    notch_mul1.operation = "MULTIPLY"
    notch_mul1.location = (-220, -640)
    links.new(z_bell.outputs["Value"], notch_mul1.inputs[0])
    links.new(notch_mask_map.outputs["Result"], notch_mul1.inputs[1])

    notch_disp = nodes.new("ShaderNodeMath")
    notch_disp.operation = "MULTIPLY"
    notch_disp.location = (-40, -640)
    links.new(notch_mul1.outputs["Value"], notch_disp.inputs[0])
    links.new(in_node.outputs[notch_sock.name], notch_disp.inputs[1])

    notch_neg = nodes.new("ShaderNodeMath")
    notch_neg.operation = "MULTIPLY"
    notch_neg.location = (120, -640)
    notch_neg.inputs[1].default_value = -1.0
    links.new(notch_disp.outputs["Value"], notch_neg.inputs[0])

    # --- 3. Medium Voronoi Fractured Blocks & Crack Lines ---
    warp_v_noise = nodes.new("ShaderNodeTexNoise")
    warp_v_noise.location = (-1080, -980)
    warp_v_noise.inputs["Scale"].default_value = 0.16 if is_stack else 0.48
    warp_v_noise.inputs["Detail"].default_value = 3.0
    links.new(seeded_pos.outputs["Vector"], warp_v_noise.inputs["Vector"])

    warp_v_vec = nodes.new("ShaderNodeVectorMath")
    warp_v_vec.operation = "SCALE"
    warp_v_vec.location = (-880, -980)
    warp_v_vec.inputs["Scale"].default_value = 0.95 if is_stack else 0.22
    links.new(warp_v_noise.outputs["Color"], warp_v_vec.inputs[0])

    warped_v_pos = nodes.new("ShaderNodeVectorMath")
    warped_v_pos.operation = "ADD"
    warped_v_pos.location = (-680, -980)
    links.new(seeded_pos.outputs["Vector"], warped_v_pos.inputs[0])
    links.new(warp_v_vec.outputs["Vector"], warped_v_pos.inputs[1])

    # Primary steep/vertical joint fractures (rotated so not axis-aligned)
    v_rot1 = nodes.new("ShaderNodeVectorRotate")
    v_rot1.rotation_type = "EULER_XYZ"
    v_rot1.location = (-500, -980)
    v_rot1.inputs["Rotation"].default_value = (0.14, -0.18, 0.35)
    links.new(warped_v_pos.outputs["Vector"], v_rot1.inputs["Vector"])

    v_scale1 = nodes.new("ShaderNodeVectorMath")
    v_scale1.operation = "MULTIPLY"
    v_scale1.location = (-320, -980)
    v_scale1.inputs[1].default_value = (0.095, 0.095, 0.030) if is_stack else (0.28, 0.28, 0.24)
    links.new(v_rot1.outputs["Vector"], v_scale1.inputs[0])

    vor_edge1 = nodes.new("ShaderNodeTexVoronoi")
    vor_edge1.feature = "DISTANCE_TO_EDGE"
    vor_edge1.location = (-120, -980)
    vor_edge1.inputs["Scale"].default_value = 1.0
    vor_edge1.inputs["Randomness"].default_value = 0.88
    links.new(v_scale1.outputs["Vector"], vor_edge1.inputs["Vector"])

    crack_disp1 = nodes.new("ShaderNodeMapRange")
    crack_disp1.location = (80, -920)
    crack_disp1.clamp = True
    crack_disp1.inputs["From Min"].default_value = 0.0
    crack_disp1.inputs["From Max"].default_value = 0.048
    crack_disp1.inputs["To Min"].default_value = -0.72 if is_stack else -0.18
    crack_disp1.inputs["To Max"].default_value = 0.0
    links.new(vor_edge1.outputs["Distance"], crack_disp1.inputs["Value"])

    crack_mask1 = nodes.new("ShaderNodeMapRange")
    crack_mask1.location = (80, -1120)
    crack_mask1.clamp = True
    crack_mask1.inputs["From Min"].default_value = 0.0
    crack_mask1.inputs["From Max"].default_value = 0.052
    crack_mask1.inputs["To Min"].default_value = 1.0
    crack_mask1.inputs["To Max"].default_value = 0.0
    links.new(vor_edge1.outputs["Distance"], crack_mask1.inputs["Value"])

    # Secondary cross-fractures + subtle F1 rock block shift
    v_rot2 = nodes.new("ShaderNodeVectorRotate")
    v_rot2.rotation_type = "EULER_XYZ"
    v_rot2.location = (-500, -1320)
    v_rot2.inputs["Rotation"].default_value = (-0.24, 0.30, -0.45)
    links.new(warped_v_pos.outputs["Vector"], v_rot2.inputs["Vector"])

    v_scale2 = nodes.new("ShaderNodeVectorMath")
    v_scale2.operation = "MULTIPLY"
    v_scale2.location = (-320, -1320)
    v_scale2.inputs[1].default_value = (0.17, 0.17, 0.11) if is_stack else (0.50, 0.50, 0.45)
    links.new(v_rot2.outputs["Vector"], v_scale2.inputs[0])

    vor_edge2 = nodes.new("ShaderNodeTexVoronoi")
    vor_edge2.feature = "DISTANCE_TO_EDGE"
    vor_edge2.location = (-120, -1320)
    vor_edge2.inputs["Scale"].default_value = 1.0
    vor_edge2.inputs["Randomness"].default_value = 0.90
    links.new(v_scale2.outputs["Vector"], vor_edge2.inputs["Vector"])

    vor_f1 = nodes.new("ShaderNodeTexVoronoi")
    vor_f1.feature = "F1"
    vor_f1.location = (-120, -1580)
    vor_f1.inputs["Scale"].default_value = 1.0
    vor_f1.inputs["Randomness"].default_value = 0.90
    links.new(v_scale2.outputs["Vector"], vor_f1.inputs["Vector"])

    crack_disp2 = nodes.new("ShaderNodeMapRange")
    crack_disp2.location = (80, -1320)
    crack_disp2.clamp = True
    crack_disp2.inputs["From Min"].default_value = 0.0
    crack_disp2.inputs["From Max"].default_value = 0.042
    crack_disp2.inputs["To Min"].default_value = -0.36 if is_stack else -0.10
    crack_disp2.inputs["To Max"].default_value = 0.0
    links.new(vor_edge2.outputs["Distance"], crack_disp2.inputs["Value"])

    crack_mask2 = nodes.new("ShaderNodeMapRange")
    crack_mask2.location = (80, -1520)
    crack_mask2.clamp = True
    crack_mask2.inputs["From Min"].default_value = 0.0
    crack_mask2.inputs["From Max"].default_value = 0.048
    crack_mask2.inputs["To Min"].default_value = 0.80
    crack_mask2.inputs["To Max"].default_value = 0.0
    links.new(vor_edge2.outputs["Distance"], crack_mask2.inputs["Value"])

    # Subtle per-block stepped offset (avoids puffy cobblestone look)
    sep_col = nodes.new("FunctionNodeSeparateColor")
    sep_col.location = (80, -1720)
    links.new(vor_f1.outputs["Color"], sep_col.inputs["Color"])

    block_height = nodes.new("ShaderNodeMapRange")
    block_height.location = (260, -1720)
    block_height.inputs["From Min"].default_value = 0.0
    block_height.inputs["From Max"].default_value = 1.0
    block_height.inputs["To Min"].default_value = -0.16 if is_stack else -0.04
    block_height.inputs["To Max"].default_value = 0.16 if is_stack else 0.04
    links.new(sep_col.outputs["Red"], block_height.inputs["Value"])

    edge_taper = nodes.new("ShaderNodeMapRange")
    edge_taper.location = (260, -1920)
    edge_taper.clamp = True
    edge_taper.inputs["From Min"].default_value = 0.02
    edge_taper.inputs["From Max"].default_value = 0.16
    edge_taper.inputs["To Min"].default_value = 0.0
    edge_taper.inputs["To Max"].default_value = 1.0
    links.new(vor_edge2.outputs["Distance"], edge_taper.inputs["Value"])

    block_disp = nodes.new("ShaderNodeMath")
    block_disp.operation = "MULTIPLY"
    block_disp.location = (460, -1760)
    links.new(block_height.outputs["Result"], block_disp.inputs[0])
    links.new(edge_taper.outputs["Result"], block_disp.inputs[1])

    # Combined crack_mask attribute
    crack_max = nodes.new("ShaderNodeMath")
    crack_max.operation = "MAXIMUM"
    crack_max.location = (320, -1240)
    links.new(crack_mask1.outputs["Result"], crack_max.inputs[0])
    links.new(crack_mask2.outputs["Result"], crack_max.inputs[1])

    # --- 4. Subtle, Tilted, Broken 3D Strata (no Z-only pancake rings) ---
    strata_rot = nodes.new("ShaderNodeVectorRotate")
    strata_rot.rotation_type = "EULER_XYZ"
    strata_rot.location = (-1080, -2100)
    links.new(seeded_pos.outputs["Vector"], strata_rot.inputs["Vector"])
    links.new(in_node.outputs[tilt_sock.name], strata_rot.inputs["Rotation"])

    strata_warp = nodes.new("ShaderNodeTexNoise")
    strata_warp.location = (-860, -2250)
    strata_warp.inputs["Scale"].default_value = 0.08
    strata_warp.inputs["Detail"].default_value = 2.5
    links.new(seeded_pos.outputs["Vector"], strata_warp.inputs["Vector"])

    strata_warp_vec = nodes.new("ShaderNodeVectorMath")
    strata_warp_vec.operation = "SCALE"
    strata_warp_vec.location = (-660, -2250)
    strata_warp_vec.inputs["Scale"].default_value = 1.35 if is_stack else 0.4
    links.new(strata_warp.outputs["Color"], strata_warp_vec.inputs[0])

    strata_pos_w = nodes.new("ShaderNodeVectorMath")
    strata_pos_w.operation = "ADD"
    strata_pos_w.location = (-460, -2140)
    links.new(strata_rot.outputs["Vector"], strata_pos_w.inputs[0])
    links.new(strata_warp_vec.outputs["Vector"], strata_pos_w.inputs[1])

    strata_aniso = nodes.new("ShaderNodeVectorMath")
    strata_aniso.operation = "MULTIPLY"
    strata_aniso.location = (-260, -2140)
    strata_aniso.inputs[1].default_value = (0.055, 0.055, 0.34) if is_stack else (0.18, 0.18, 0.45)
    links.new(strata_pos_w.outputs["Vector"], strata_aniso.inputs[0])

    strata_noise_3d = nodes.new("ShaderNodeTexNoise")
    strata_noise_3d.location = (-60, -2140)
    strata_noise_3d.inputs["Scale"].default_value = 1.0
    strata_noise_3d.inputs["Detail"].default_value = 4.0
    strata_noise_3d.inputs["Roughness"].default_value = 0.58
    links.new(strata_aniso.outputs["Vector"], strata_noise_3d.inputs["Vector"])

    strata_map = nodes.new("ShaderNodeMapRange")
    strata_map.location = (140, -2140)
    strata_map.inputs["From Min"].default_value = 0.34
    strata_map.inputs["From Max"].default_value = 0.66
    strata_map.inputs["To Min"].default_value = -0.18 if is_stack else -0.04
    strata_map.inputs["To Max"].default_value = 0.18 if is_stack else 0.04
    links.new(strata_noise_3d.outputs["Fac"], strata_map.inputs["Value"])

    # Breakup mask so strata pinches out across sheer faces
    strata_break = nodes.new("ShaderNodeTexNoise")
    strata_break.location = (-60, -2380)
    strata_break.inputs["Scale"].default_value = 0.13
    strata_break.inputs["Detail"].default_value = 2.0
    links.new(seeded_pos.outputs["Vector"], strata_break.inputs["Vector"])

    strata_break_map = nodes.new("ShaderNodeMapRange")
    strata_break_map.location = (140, -2380)
    strata_break_map.clamp = True
    strata_break_map.inputs["From Min"].default_value = 0.32
    strata_break_map.inputs["From Max"].default_value = 0.68
    strata_break_map.inputs["To Min"].default_value = 0.15
    strata_break_map.inputs["To Max"].default_value = 1.0
    links.new(strata_break.outputs["Fac"], strata_break_map.inputs["Value"])

    strata_disp = nodes.new("ShaderNodeMath")
    strata_disp.operation = "MULTIPLY"
    strata_disp.location = (360, -2200)
    links.new(strata_map.outputs["Result"], strata_disp.inputs[0])
    links.new(strata_break_map.outputs["Result"], strata_disp.inputs[1])

    # --- 5. Medium Crags & Fine Rock Grain ---
    grain_noise = nodes.new("ShaderNodeTexNoise")
    grain_noise.location = (-60, -2620)
    grain_noise.inputs["Scale"].default_value = 0.24 if is_stack else 0.95
    grain_noise.inputs["Detail"].default_value = 6.0
    grain_noise.inputs["Roughness"].default_value = 0.66
    links.new(seeded_pos.outputs["Vector"], grain_noise.inputs["Vector"])

    grain_disp = nodes.new("ShaderNodeMapRange")
    grain_disp.location = (160, -2620)
    grain_disp.inputs["From Min"].default_value = 0.25
    grain_disp.inputs["From Max"].default_value = 0.75
    grain_disp.inputs["To Min"].default_value = -0.34 if is_stack else -0.07
    grain_disp.inputs["To Max"].default_value = 0.34 if is_stack else 0.07
    links.new(grain_noise.outputs["Fac"], grain_disp.inputs["Value"])

    # Sum all scalar normal displacements
    def _add_scalar(loc, a_sock, b_sock):
        n = nodes.new("ShaderNodeMath")
        n.operation = "ADD"
        n.location = loc
        links.new(a_sock, n.inputs[0])
        links.new(b_sock, n.inputs[1])
        return n.outputs["Value"]

    s1 = _add_scalar((420, -760), notch_neg.outputs["Value"], crack_disp1.outputs["Result"])
    s2 = _add_scalar((580, -1000), s1, crack_disp2.outputs["Result"])
    s3 = _add_scalar((660, -1400), s2, block_disp.outputs["Value"])
    s4 = _add_scalar((760, -1750), s3, strata_disp.outputs["Value"])
    s_total = _add_scalar((860, -2050), s4, grain_disp.outputs["Result"])

    normal_disp = nodes.new("ShaderNodeVectorMath")
    normal_disp.operation = "SCALE"
    normal_disp.location = (820, -320)
    links.new(norm_node.outputs["Normal"], normal_disp.inputs[0])
    links.new(s_total, normal_disp.inputs["Scale"])

    total_offset = nodes.new("ShaderNodeVectorMath")
    total_offset.operation = "ADD"
    total_offset.location = (980, -200)
    links.new(macro_vec.outputs["Vector"], total_offset.inputs[0])
    links.new(normal_disp.outputs["Vector"], total_offset.inputs[1])

    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (820, 0)
    geo_out_src = prep_geo.outputs["Geometry"] if is_stack else prep_geo.outputs["Mesh"]
    links.new(geo_out_src, set_pos.inputs["Geometry"])
    links.new(total_offset.outputs["Vector"], set_pos.inputs["Offset"])

    # Store crack_mask for shader crevice darkening
    store_crack = nodes.new("GeometryNodeStoreNamedAttribute")
    store_crack.data_type = "FLOAT"
    store_crack.domain = "POINT"
    store_crack.location = (1000, 0)
    store_crack.inputs["Name"].default_value = "crack_mask"
    links.new(set_pos.outputs["Geometry"], store_crack.inputs["Geometry"])
    links.new(crack_max.outputs["Value"], store_crack.inputs["Value"])

    # Smooth faces, keep sharp dihedral fracture ridges crisp
    shade_faces = nodes.new("GeometryNodeSetShadeSmooth")
    shade_faces.domain = "FACE"
    shade_faces.location = (1160, 0)
    shade_faces.inputs["Shade Smooth"].default_value = True
    links.new(store_crack.outputs["Geometry"], shade_faces.inputs["Geometry"])

    edge_angle = nodes.new("GeometryNodeInputMeshEdgeAngle")
    edge_angle.location = (1000, 220)

    sharp_cmp = nodes.new("ShaderNodeMath")
    sharp_cmp.operation = "GREATER_THAN"
    sharp_cmp.location = (1160, 220)
    sharp_cmp.inputs[1].default_value = 0.68 if is_stack else 0.56
    links.new(edge_angle.outputs["Unsigned Angle"], sharp_cmp.inputs[0])

    shade_edges = nodes.new("GeometryNodeSetShadeSmooth")
    shade_edges.domain = "EDGE"
    shade_edges.location = (1330, 0)
    shade_edges.inputs["Shade Smooth"].default_value = False
    links.new(shade_faces.outputs["Geometry"], shade_edges.inputs["Geometry"])
    links.new(sharp_cmp.outputs["Value"], shade_edges.inputs["Selection"])
    links.new(shade_edges.outputs["Geometry"], out_node.inputs["Geometry"])

    return ng, seed_sock, tilt_sock, notch_sock


def _create_rock_material():
    """Build procedural rock shader with per-stack color, cracks, grass caps, algae, guano, and wetness."""
    mat = bpy.data.materials.new(name=f"{NAME_PREFIX}RockMat")
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1800, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1450, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    geom = nodes.new("ShaderNodeNewGeometry")
    geom.location = (-2100, 200)

    obj_info = nodes.new("ShaderNodeObjectInfo")
    obj_info.location = (-2100, 560)

    sep_world = nodes.new("ShaderNodeSeparateXYZ")
    sep_world.location = (-1820, 200)
    links.new(geom.outputs["Position"], sep_world.inputs["Vector"])

    sep_norm = nodes.new("ShaderNodeSeparateXYZ")
    sep_norm.location = (-1820, -20)
    links.new(geom.outputs["Normal"], sep_norm.inputs["Vector"])

    # Per-object coordinate offset from Object Info -> Random
    obj_off_mul = nodes.new("ShaderNodeMath")
    obj_off_mul.operation = "MULTIPLY"
    obj_off_mul.location = (-1820, 560)
    obj_off_mul.inputs[1].default_value = 173.7
    links.new(obj_info.outputs["Random"], obj_off_mul.inputs[0])

    obj_off_vec = nodes.new("ShaderNodeCombineXYZ")
    obj_off_vec.location = (-1620, 560)
    links.new(obj_off_mul.outputs["Value"], obj_off_vec.inputs["X"])
    links.new(obj_off_mul.outputs["Value"], obj_off_vec.inputs["Y"])

    rock_pos = nodes.new("ShaderNodeVectorMath")
    rock_pos.operation = "ADD"
    rock_pos.location = (-1420, 460)
    links.new(geom.outputs["Position"], rock_pos.inputs[0])
    links.new(obj_off_vec.outputs["Vector"], rock_pos.inputs[1])

    # --- 1. Subtle Tilted Broken Strata + Patchy Rock Variation ---
    strata_rot = nodes.new("ShaderNodeVectorRotate")
    strata_rot.rotation_type = "EULER_XYZ"
    strata_rot.location = (-1220, 620)
    strata_rot.inputs["Rotation"].default_value = (0.16, -0.10, 0.0)
    links.new(rock_pos.outputs["Vector"], strata_rot.inputs["Vector"])

    strata_warp = nodes.new("ShaderNodeTexNoise")
    strata_warp.location = (-1220, 840)
    strata_warp.inputs["Scale"].default_value = 0.09
    strata_warp.inputs["Detail"].default_value = 2.5
    links.new(rock_pos.outputs["Vector"], strata_warp.inputs["Vector"])

    strata_warp_vec = nodes.new("ShaderNodeVectorMath")
    strata_warp_vec.operation = "SCALE"
    strata_warp_vec.location = (-1020, 840)
    strata_warp_vec.inputs["Scale"].default_value = 1.5
    links.new(strata_warp.outputs["Color"], strata_warp_vec.inputs[0])

    strata_wpos = nodes.new("ShaderNodeVectorMath")
    strata_wpos.operation = "ADD"
    strata_wpos.location = (-840, 720)
    links.new(strata_rot.outputs["Vector"], strata_wpos.inputs[0])
    links.new(strata_warp_vec.outputs["Vector"], strata_wpos.inputs[1])

    strata_scale = nodes.new("ShaderNodeVectorMath")
    strata_scale.operation = "MULTIPLY"
    strata_scale.location = (-660, 720)
    strata_scale.inputs[1].default_value = (0.06, 0.06, 0.30)
    links.new(strata_wpos.outputs["Vector"], strata_scale.inputs[0])

    strata_noise = nodes.new("ShaderNodeTexNoise")
    strata_noise.location = (-460, 720)
    strata_noise.inputs["Scale"].default_value = 1.0
    strata_noise.inputs["Detail"].default_value = 5.0
    strata_noise.inputs["Roughness"].default_value = 0.62
    links.new(strata_scale.outputs["Vector"], strata_noise.inputs["Vector"])

    patch_noise = nodes.new("ShaderNodeTexNoise")
    patch_noise.location = (-660, 480)
    patch_noise.inputs["Scale"].default_value = 0.16
    patch_noise.inputs["Detail"].default_value = 5.0
    patch_noise.inputs["Roughness"].default_value = 0.60
    links.new(rock_pos.outputs["Vector"], patch_noise.inputs["Vector"])

    # Mix subtle strata (40%) with isotropic patchy rock texture (60%)
    rock_tex_mix = nodes.new("ShaderNodeMix")
    rock_tex_mix.data_type = "FLOAT"
    rock_tex_mix.location = (-240, 620)
    rock_tex_mix.inputs["Factor"].default_value = 0.58
    links.new(strata_noise.outputs["Fac"], rock_tex_mix.inputs["A"])
    links.new(patch_noise.outputs["Fac"], rock_tex_mix.inputs["B"])

    # Warm sandstone palette (Twelve Apostles style, realistic deep earthy values)
    warm_ramp = nodes.new("ShaderNodeValToRGB")
    warm_ramp.location = (-20, 760)
    wcr = warm_ramp.color_ramp
    wcr.elements[0].position = 0.22
    wcr.elements[0].color = (0.055, 0.042, 0.032, 1.0)
    wcr.elements[1].position = 0.80
    wcr.elements[1].color = (0.215, 0.148, 0.095, 1.0)
    we1 = wcr.elements.new(0.46)
    we1.color = (0.115, 0.088, 0.065, 1.0)
    we2 = wcr.elements.new(0.64)
    we2.color = (0.165, 0.112, 0.072, 1.0)
    links.new(rock_tex_mix.outputs["Result"], warm_ramp.inputs["Fac"])

    # Cool grey-brown / coastal basalt-slate palette (Faroe style)
    cool_ramp = nodes.new("ShaderNodeValToRGB")
    cool_ramp.location = (-20, 480)
    ccr = cool_ramp.color_ramp
    ccr.elements[0].position = 0.24
    ccr.elements[0].color = (0.024, 0.027, 0.033, 1.0)
    ccr.elements[1].position = 0.78
    ccr.elements[1].color = (0.115, 0.108, 0.102, 1.0)
    ce1 = ccr.elements.new(0.50)
    ce1.color = (0.066, 0.063, 0.060, 1.0)
    links.new(rock_tex_mix.outputs["Result"], cool_ramp.inputs["Fac"])

    # Per-object blend between warm sandstone and grey-brown rock via custom attribute
    attr_tone = nodes.new("ShaderNodeAttribute")
    attr_tone.attribute_type = "OBJECT"
    attr_tone.attribute_name = "rock_tone"
    attr_tone.location = (-20, 960)

    mix_palette = nodes.new("ShaderNodeMix")
    mix_palette.data_type = "RGBA"
    mix_palette.location = (260, 620)
    links.new(attr_tone.outputs["Fac"], mix_palette.inputs["Factor"])
    links.new(cool_ramp.outputs["Color"], mix_palette.inputs["A"])
    links.new(warm_ramp.outputs["Color"], mix_palette.inputs["B"])

    # --- 2. Crack Darkening from GeoNodes "crack_mask" + Fine Shader Voronoi ---
    attr_crack = nodes.new("ShaderNodeAttribute")
    attr_crack.attribute_name = "crack_mask"
    attr_crack.location = (-460, 220)

    fine_crack_warp = nodes.new("ShaderNodeTexNoise")
    fine_crack_warp.location = (-660, 20)
    fine_crack_warp.inputs["Scale"].default_value = 0.65
    fine_crack_warp.inputs["Detail"].default_value = 3.0
    links.new(rock_pos.outputs["Vector"], fine_crack_warp.inputs["Vector"])

    fine_crack_wvec = nodes.new("ShaderNodeVectorMath")
    fine_crack_wvec.operation = "SCALE"
    fine_crack_wvec.location = (-460, 20)
    fine_crack_wvec.inputs["Scale"].default_value = 0.75
    links.new(fine_crack_warp.outputs["Color"], fine_crack_wvec.inputs[0])

    fine_crack_pos = nodes.new("ShaderNodeVectorMath")
    fine_crack_pos.operation = "ADD"
    fine_crack_pos.location = (-280, 20)
    links.new(rock_pos.outputs["Vector"], fine_crack_pos.inputs[0])
    links.new(fine_crack_wvec.outputs["Vector"], fine_crack_pos.inputs[1])

    fine_vor = nodes.new("ShaderNodeTexVoronoi")
    fine_vor.feature = "DISTANCE_TO_EDGE"
    fine_vor.location = (-80, 20)
    fine_vor.inputs["Scale"].default_value = 0.28
    fine_vor.inputs["Randomness"].default_value = 0.92
    links.new(fine_crack_pos.outputs["Vector"], fine_vor.inputs["Vector"])

    fine_crack_map = nodes.new("ShaderNodeMapRange")
    fine_crack_map.location = (120, 20)
    fine_crack_map.clamp = True
    fine_crack_map.inputs["From Min"].default_value = 0.0
    fine_crack_map.inputs["From Max"].default_value = 0.032
    fine_crack_map.inputs["To Min"].default_value = 0.62
    fine_crack_map.inputs["To Max"].default_value = 0.0
    links.new(fine_vor.outputs["Distance"], fine_crack_map.inputs["Value"])

    # Break up crack continuity so fissures pinch open and closed organically
    crack_break_noise = nodes.new("ShaderNodeTexNoise")
    crack_break_noise.location = (120, -180)
    crack_break_noise.inputs["Scale"].default_value = 0.60
    crack_break_noise.inputs["Detail"].default_value = 3.0
    links.new(rock_pos.outputs["Vector"], crack_break_noise.inputs["Vector"])

    crack_break_map = nodes.new("ShaderNodeMapRange")
    crack_break_map.location = (300, -180)
    crack_break_map.clamp = True
    crack_break_map.inputs["From Min"].default_value = 0.32
    crack_break_map.inputs["From Max"].default_value = 0.68
    crack_break_map.inputs["To Min"].default_value = 0.20
    crack_break_map.inputs["To Max"].default_value = 1.0
    links.new(crack_break_noise.outputs["Fac"], crack_break_map.inputs["Value"])

    crack_combine_raw = nodes.new("ShaderNodeMath")
    crack_combine_raw.operation = "MAXIMUM"
    crack_combine_raw.location = (320, 140)
    links.new(attr_crack.outputs["Fac"], crack_combine_raw.inputs[0])
    links.new(fine_crack_map.outputs["Result"], crack_combine_raw.inputs[1])

    crack_combine = nodes.new("ShaderNodeMath")
    crack_combine.operation = "MULTIPLY"
    crack_combine.location = (440, 60)
    links.new(crack_combine_raw.outputs["Value"], crack_combine.inputs[0])
    links.new(crack_break_map.outputs["Result"], crack_combine.inputs[1])

    crack_darken_fac = nodes.new("ShaderNodeMath")
    crack_darken_fac.operation = "MULTIPLY"
    crack_darken_fac.location = (480, 180)
    crack_darken_fac.inputs[1].default_value = 0.82
    links.new(crack_combine.outputs["Value"], crack_darken_fac.inputs[0])

    mix_cracks = nodes.new("ShaderNodeMix")
    mix_cracks.data_type = "RGBA"
    mix_cracks.location = (560, 560)
    mix_cracks.inputs["B"].default_value = (0.018, 0.016, 0.014, 1.0)
    links.new(crack_darken_fac.outputs["Value"], mix_cracks.inputs["Factor"])
    links.new(mix_palette.outputs["Result"], mix_cracks.inputs["A"])

    # --- 3. Bird Guano Streaks on Upper Ledges ---
    guano_norm = nodes.new("ShaderNodeMapRange")
    guano_norm.location = (-840, -180)
    guano_norm.clamp = True
    guano_norm.inputs["From Min"].default_value = 0.22
    guano_norm.inputs["From Max"].default_value = 0.62
    links.new(sep_norm.outputs["Z"], guano_norm.inputs["Value"])

    guano_z = nodes.new("ShaderNodeMapRange")
    guano_z.location = (-840, -380)
    guano_z.clamp = True
    guano_z.inputs["From Min"].default_value = 7.5
    guano_z.inputs["From Max"].default_value = 13.0
    links.new(sep_world.outputs["Z"], guano_z.inputs["Value"])

    streak_coord = nodes.new("ShaderNodeVectorMath")
    streak_coord.operation = "MULTIPLY"
    streak_coord.location = (-1020, -560)
    streak_coord.inputs[1].default_value = (1.35, 1.35, 0.065)
    links.new(geom.outputs["Position"], streak_coord.inputs[0])

    streak_noise = nodes.new("ShaderNodeTexNoise")
    streak_noise.location = (-840, -560)
    streak_noise.inputs["Scale"].default_value = 0.85
    streak_noise.inputs["Detail"].default_value = 5.0
    streak_noise.inputs["Roughness"].default_value = 0.65
    links.new(streak_coord.outputs["Vector"], streak_noise.inputs["Vector"])

    streak_ramp = nodes.new("ShaderNodeValToRGB")
    streak_ramp.location = (-640, -560)
    streak_ramp.color_ramp.elements[0].position = 0.48
    streak_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    streak_ramp.color_ramp.elements[1].position = 0.68
    streak_ramp.color_ramp.elements[1].color = (0.75, 0.75, 0.75, 1.0)
    links.new(streak_noise.outputs["Fac"], streak_ramp.inputs["Fac"])

    g_mul1 = nodes.new("ShaderNodeMath")
    g_mul1.operation = "MULTIPLY"
    g_mul1.location = (-460, -280)
    links.new(guano_norm.outputs["Result"], g_mul1.inputs[0])
    links.new(guano_z.outputs["Result"], g_mul1.inputs[1])

    guano_mask = nodes.new("ShaderNodeMath")
    guano_mask.operation = "MULTIPLY"
    guano_mask.location = (-280, -340)
    links.new(g_mul1.outputs["Value"], guano_mask.inputs[0])
    links.new(streak_ramp.outputs["Color"], guano_mask.inputs[1])

    mix_guano = nodes.new("ShaderNodeMix")
    mix_guano.data_type = "RGBA"
    mix_guano.location = (740, 520)
    mix_guano.inputs["B"].default_value = (0.60, 0.57, 0.50, 1.0)
    links.new(guano_mask.outputs["Value"], mix_guano.inputs["Factor"])
    links.new(mix_cracks.outputs["Result"], mix_guano.inputs["A"])

    # --- 4. Windswept Coastal Grass on Some Fractured Stack Tops ---
    grass_norm = nodes.new("ShaderNodeMapRange")
    grass_norm.location = (-840, -800)
    grass_norm.clamp = True
    grass_norm.inputs["From Min"].default_value = 0.45
    grass_norm.inputs["From Max"].default_value = 0.68
    links.new(sep_norm.outputs["Z"], grass_norm.inputs["Value"])

    grass_z = nodes.new("ShaderNodeMapRange")
    grass_z.location = (-840, -1000)
    grass_z.clamp = True
    grass_z.inputs["From Min"].default_value = 9.5
    grass_z.inputs["From Max"].default_value = 14.5
    links.new(sep_world.outputs["Z"], grass_z.inputs["Value"])

    grass_obj_enable = nodes.new("ShaderNodeAttribute")
    grass_obj_enable.attribute_type = "OBJECT"
    grass_obj_enable.attribute_name = "has_grass"
    grass_obj_enable.location = (-840, -1180)

    grass_edge_noise = nodes.new("ShaderNodeTexNoise")
    grass_edge_noise.location = (-1040, -1340)
    grass_edge_noise.inputs["Scale"].default_value = 0.55
    grass_edge_noise.inputs["Detail"].default_value = 5.0
    links.new(rock_pos.outputs["Vector"], grass_edge_noise.inputs["Vector"])

    grass_edge_ramp = nodes.new("ShaderNodeMapRange")
    grass_edge_ramp.location = (-840, -1340)
    grass_edge_ramp.clamp = True
    grass_edge_ramp.inputs["From Min"].default_value = 0.34
    grass_edge_ramp.inputs["From Max"].default_value = 0.62
    links.new(grass_edge_noise.outputs["Fac"], grass_edge_ramp.inputs["Value"])

    gr_m1 = nodes.new("ShaderNodeMath")
    gr_m1.operation = "MULTIPLY"
    gr_m1.location = (-600, -900)
    links.new(grass_norm.outputs["Result"], gr_m1.inputs[0])
    links.new(grass_z.outputs["Result"], gr_m1.inputs[1])

    gr_m2 = nodes.new("ShaderNodeMath")
    gr_m2.operation = "MULTIPLY"
    gr_m2.location = (-420, -1020)
    links.new(gr_m1.outputs["Value"], gr_m2.inputs[0])
    links.new(grass_obj_enable.outputs["Fac"], gr_m2.inputs[1])

    grass_mask = nodes.new("ShaderNodeMath")
    grass_mask.operation = "MULTIPLY"
    grass_mask.location = (-240, -1100)
    links.new(gr_m2.outputs["Value"], grass_mask.inputs[0])
    links.new(grass_edge_ramp.outputs["Result"], grass_mask.inputs[1])

    grass_col_noise = nodes.new("ShaderNodeTexNoise")
    grass_col_noise.location = (-240, -1320)
    grass_col_noise.inputs["Scale"].default_value = 3.2
    grass_col_noise.inputs["Detail"].default_value = 6.0
    links.new(geom.outputs["Position"], grass_col_noise.inputs["Vector"])

    grass_ramp = nodes.new("ShaderNodeValToRGB")
    grass_ramp.location = (0, -1320)
    grass_ramp.color_ramp.elements[0].position = 0.30
    grass_ramp.color_ramp.elements[0].color = (0.055, 0.095, 0.028, 1.0)
    grass_ramp.color_ramp.elements[1].position = 0.75
    grass_ramp.color_ramp.elements[1].color = (0.145, 0.155, 0.052, 1.0)
    links.new(grass_col_noise.outputs["Fac"], grass_ramp.inputs["Fac"])

    mix_grass = nodes.new("ShaderNodeMix")
    mix_grass.data_type = "RGBA"
    mix_grass.location = (920, 480)
    links.new(grass_mask.outputs["Value"], mix_grass.inputs["Factor"])
    links.new(mix_guano.outputs["Result"], mix_grass.inputs["A"])
    links.new(grass_ramp.outputs["Color"], mix_grass.inputs["B"])

    # --- 5. Waterline Algae Band & Wet Base (z < 1.8, strongest < 1.2) ---
    splash_noise = nodes.new("ShaderNodeTexNoise")
    splash_noise.location = (-1600, -1600)
    splash_noise.inputs["Scale"].default_value = 0.38
    splash_noise.inputs["Detail"].default_value = 4.0
    links.new(geom.outputs["Position"], splash_noise.inputs["Vector"])

    z_noisy = nodes.new("ShaderNodeMath")
    z_noisy.operation = "MULTIPLY_ADD"
    z_noisy.location = (-1380, -1600)
    z_noisy.inputs[1].default_value = 0.50
    links.new(splash_noise.outputs["Fac"], z_noisy.inputs[0])
    links.new(sep_world.outputs["Z"], z_noisy.inputs[2])

    band_low = nodes.new("ShaderNodeMapRange")
    band_low.location = (-1160, -1600)
    band_low.clamp = True
    band_low.inputs["From Min"].default_value = -1.0
    band_low.inputs["From Max"].default_value = -0.2
    links.new(z_noisy.outputs["Value"], band_low.inputs["Value"])

    band_high = nodes.new("ShaderNodeMapRange")
    band_high.location = (-1160, -1800)
    band_high.clamp = True
    band_high.inputs["From Min"].default_value = 0.55
    band_high.inputs["From Max"].default_value = 1.35
    band_high.inputs["To Min"].default_value = 1.0
    band_high.inputs["To Max"].default_value = 0.0
    links.new(z_noisy.outputs["Value"], band_high.inputs["Value"])

    waterline_band = nodes.new("ShaderNodeMath")
    waterline_band.operation = "MULTIPLY"
    waterline_band.location = (-940, -1700)
    links.new(band_low.outputs["Result"], waterline_band.inputs[0])
    links.new(band_high.outputs["Result"], waterline_band.inputs[1])

    algae_noise = nodes.new("ShaderNodeTexNoise")
    algae_noise.location = (-940, -1900)
    algae_noise.inputs["Scale"].default_value = 1.5
    algae_noise.inputs["Detail"].default_value = 5.0
    links.new(geom.outputs["Position"], algae_noise.inputs["Vector"])

    algae_ramp = nodes.new("ShaderNodeMapRange")
    algae_ramp.location = (-720, -1900)
    algae_ramp.clamp = True
    algae_ramp.inputs["From Min"].default_value = 0.35
    algae_ramp.inputs["From Max"].default_value = 0.72
    links.new(algae_noise.outputs["Fac"], algae_ramp.inputs["Value"])

    algae_mask = nodes.new("ShaderNodeMath")
    algae_mask.operation = "MULTIPLY"
    algae_mask.location = (-500, -1760)
    links.new(waterline_band.outputs["Value"], algae_mask.inputs[0])
    links.new(algae_ramp.outputs["Result"], algae_mask.inputs[1])

    mix_algae = nodes.new("ShaderNodeMix")
    mix_algae.data_type = "RGBA"
    mix_algae.location = (1080, 420)
    mix_algae.inputs["B"].default_value = (0.022, 0.048, 0.016, 1.0)
    links.new(algae_mask.outputs["Value"], mix_algae.inputs["Factor"])
    links.new(mix_grass.outputs["Result"], mix_algae.inputs["A"])

    # Wet darkening (strongest below z=1.15, fades out by z=2.1)
    wet_map = nodes.new("ShaderNodeMapRange")
    wet_map.location = (-940, -1480)
    wet_map.clamp = True
    wet_map.inputs["From Min"].default_value = 1.15
    wet_map.inputs["From Max"].default_value = 2.15
    wet_map.inputs["To Min"].default_value = 1.0
    wet_map.inputs["To Max"].default_value = 0.0
    links.new(z_noisy.outputs["Value"], wet_map.inputs["Value"])

    wet_darken = nodes.new("ShaderNodeHueSaturation")
    wet_darken.location = (1260, 380)
    wet_darken.inputs["Value"].default_value = 0.30
    wet_darken.inputs["Saturation"].default_value = 1.12
    links.new(wet_map.outputs["Result"], wet_darken.inputs["Fac"])
    links.new(mix_algae.outputs["Result"], wet_darken.inputs["Color"])
    links.new(wet_darken.outputs["Color"], bsdf.inputs["Base Color"])

    # --- 6. Roughness & Wet Coat (keep wet stone textured, not glossy plastic) ---
    rough_noise = nodes.new("ShaderNodeTexNoise")
    rough_noise.location = (260, -280)
    rough_noise.inputs["Scale"].default_value = 3.8
    rough_noise.inputs["Detail"].default_value = 6.0
    links.new(geom.outputs["Position"], rough_noise.inputs["Vector"])

    dry_rough = nodes.new("ShaderNodeMapRange")
    dry_rough.location = (480, -280)
    dry_rough.inputs["From Min"].default_value = 0.25
    dry_rough.inputs["From Max"].default_value = 0.75
    dry_rough.inputs["To Min"].default_value = 0.72
    dry_rough.inputs["To Max"].default_value = 0.94
    links.new(rough_noise.outputs["Fac"], dry_rough.inputs["Value"])

    wet_rough = nodes.new("ShaderNodeMapRange")
    wet_rough.location = (480, -480)
    wet_rough.inputs["From Min"].default_value = 0.25
    wet_rough.inputs["From Max"].default_value = 0.75
    wet_rough.inputs["To Min"].default_value = 0.14
    wet_rough.inputs["To Max"].default_value = 0.34
    links.new(rough_noise.outputs["Fac"], wet_rough.inputs["Value"])

    mix_wet_rough = nodes.new("ShaderNodeMix")
    mix_wet_rough.data_type = "FLOAT"
    mix_wet_rough.location = (740, -340)
    links.new(wet_map.outputs["Result"], mix_wet_rough.inputs["Factor"])
    links.new(dry_rough.outputs["Result"], mix_wet_rough.inputs["A"])
    links.new(wet_rough.outputs["Result"], mix_wet_rough.inputs["B"])

    # Cracks and grass are always matte
    matte_mask = nodes.new("ShaderNodeMath")
    matte_mask.operation = "MAXIMUM"
    matte_mask.location = (740, -140)
    links.new(crack_combine.outputs["Value"], matte_mask.inputs[0])
    links.new(grass_mask.outputs["Value"], matte_mask.inputs[1])

    mix_final_rough = nodes.new("ShaderNodeMix")
    mix_final_rough.data_type = "FLOAT"
    mix_final_rough.location = (1040, -220)
    mix_final_rough.inputs["B"].default_value = 0.93
    links.new(matte_mask.outputs["Value"], mix_final_rough.inputs["Factor"])
    links.new(mix_wet_rough.outputs["Result"], mix_final_rough.inputs["A"])
    links.new(mix_final_rough.outputs["Result"], bsdf.inputs["Roughness"])

    coat_mul = nodes.new("ShaderNodeMath")
    coat_mul.operation = "MULTIPLY"
    coat_mul.location = (1040, -420)
    coat_mul.inputs[1].default_value = 0.30
    links.new(wet_map.outputs["Result"], coat_mul.inputs[0])
    links.new(coat_mul.outputs["Value"], bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.16

    # --- 7. Multi-Scale Rock Grain & Crack Bump ---
    micro_noise = nodes.new("ShaderNodeTexNoise")
    micro_noise.location = (260, -720)
    micro_noise.inputs["Scale"].default_value = 4.2
    micro_noise.inputs["Detail"].default_value = 8.0
    micro_noise.inputs["Roughness"].default_value = 0.70
    links.new(geom.outputs["Position"], micro_noise.inputs["Vector"])

    fine_noise = nodes.new("ShaderNodeTexNoise")
    fine_noise.location = (260, -940)
    fine_noise.inputs["Scale"].default_value = 20.0
    fine_noise.inputs["Detail"].default_value = 6.0
    fine_noise.inputs["Roughness"].default_value = 0.65
    links.new(geom.outputs["Position"], fine_noise.inputs["Vector"])

    b_add0 = nodes.new("ShaderNodeMath")
    b_add0.operation = "MULTIPLY_ADD"
    b_add0.location = (500, -820)
    b_add0.inputs[1].default_value = 0.42
    links.new(fine_noise.outputs["Fac"], b_add0.inputs[0])
    links.new(micro_noise.outputs["Fac"], b_add0.inputs[2])

    b_add1 = nodes.new("ShaderNodeMath")
    b_add1.operation = "MULTIPLY_ADD"
    b_add1.location = (700, -760)
    b_add1.inputs[1].default_value = 0.24
    links.new(strata_noise.outputs["Fac"], b_add1.inputs[0])
    links.new(b_add0.outputs["Value"], b_add1.inputs[2])

    b_crack = nodes.new("ShaderNodeMath")
    b_crack.operation = "MULTIPLY_ADD"
    b_crack.location = (900, -700)
    b_crack.inputs[1].default_value = -0.95
    links.new(crack_combine.outputs["Value"], b_crack.inputs[0])
    links.new(b_add1.outputs["Value"], b_crack.inputs[2])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (1140, -640)
    bump.inputs["Strength"].default_value = 0.80
    bump.inputs["Distance"].default_value = 0.14
    links.new(b_crack.outputs["Value"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


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

    for px, py, pr in placed:
        if math.hypot(x - px, y - py) < radius + pr + 2.2:
            return False

    return True


def _set_geonode_input(mod, sock, value):
    """Set a Geometry Nodes modifier input value using Blender 5.x properties API."""
    if hasattr(mod, "properties") and hasattr(mod.properties, "inputs"):
        sock_prop = getattr(mod.properties.inputs, sock.identifier, None)
        if sock_prop is not None:
            sock_prop.value = value
            return
    mod[sock.identifier] = value


def _create_rock_object(
    name,
    bm,
    location,
    col,
    mat,
    geo_info,
    rng,
    rock_tone=0.5,
    has_grass=0.0,
    voxel_size=None,
):
    """Create mesh object, assign material/attributes/modifiers, and link to collection."""
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj["rock_tone"] = float(rock_tone)
    obj["has_grass"] = float(has_grass)
    col.objects.link(obj)
    obj.data.materials.append(mat)

    if voxel_size is not None:
        remesh = obj.modifiers.new(name="VoxelFuse", type="REMESH")
        remesh.mode = "VOXEL"
        remesh.voxel_size = voxel_size
        remesh.use_smooth_shade = True

    geo_group, seed_sock, tilt_sock, notch_sock = geo_info
    mod = obj.modifiers.new(name="RockDetail", type="NODES")
    mod.node_group = geo_group
    _set_geonode_input(
        mod,
        seed_sock,
        (
            rng.uniform(-100.0, 100.0),
            rng.uniform(-100.0, 100.0),
            rng.uniform(-30.0, 30.0),
        ),
    )
    _set_geonode_input(
        mod,
        tilt_sock,
        (
            rng.uniform(-0.22, 0.22),
            rng.uniform(-0.22, 0.22),
            rng.uniform(-0.30, 0.30),
        ),
    )
    _set_geonode_input(mod, notch_sock, rng.uniform(0.95, 1.65) if voxel_size is not None else 0.0)

    return obj


def build():
    """Idempotently build the SeaStacks collection with asymmetric stacks and angular boulders."""
    _cleanup_existing()
    col = _get_or_create_collection()
    rng = random.Random(SEED)

    mat = _create_rock_material()
    stack_geo = _create_rock_geometry_nodes(f"{NAME_PREFIX}StackGeo", is_stack=True)
    boulder_geo = _create_rock_geometry_nodes(f"{NAME_PREFIX}BoulderGeo", is_stack=False)

    placed = []

    # 5 tall asymmetric sea stacks across left (x < -20) and right (x > 25), near (y 50-70) & far (y 140-170)
    # Tuples: (x, y, radius, height, archetype, rock_tone [0=grey-brown..1=warm sandstone], has_grass)
    stack_specs = [
        (-42.0, 62.0, 6.2, 21.5, 0, 0.85, 1.0),   # Near-Left warm sandstone stack with grass cap
        (-34.0, 152.0, 7.6, 31.0, 1, 0.22, 0.0),  # Far-Left grey-brown hero stack
        (47.0, 60.0, 6.4, 22.5, 2, 0.15, 1.0),    # Near-Right dark grey-brown stack with grass cap
        (40.0, 115.0, 7.2, 29.0, 3, 0.78, 1.0),   # Mid-Right warm sandstone stack with grass cap
        (74.0, 148.0, 8.0, 33.5, 4, 0.42, 0.0),   # Far-Right weathered grey-tan spire
    ]

    for i, (x, y, radius, height, archetype, tone, grass) in enumerate(stack_specs):
        bm = _build_stack_bmesh(rng, radius, height, archetype)
        _create_rock_object(
            name=f"{NAME_PREFIX}Stack_{i + 1:02d}",
            bm=bm,
            location=(x, y, 0.0),
            col=col,
            mat=mat,
            geo_info=stack_geo,
            rng=rng,
            rock_tone=tone,
            has_grass=grass,
            voxel_size=0.24,
        )
        placed.append((x, y, radius * 1.15))

    # 13 angular, partly submerged reef boulders (tops at z = 0.35..2.45)
    # Seed foreground/midground reef outcrops on both left and right sides
    candidate_zones = [
        (-30.0, 48.0),
        (-49.0, 52.0),
        (-25.0, 78.0),
        (-44.0, 92.0),
        (34.0, 48.0),
        (56.0, 74.0),
        (30.0, 102.0),
    ]
    boulder_target = 13
    boulder_idx = 0

    for zx, zy in candidate_zones:
        if boulder_idx >= boulder_target:
            break
        radius = rng.uniform(2.2, 4.3)
        bx = zx + rng.uniform(-3.5, 3.5)
        by = zy + rng.uniform(-3.5, 3.5)
        if _is_valid_placement(bx, by, radius, placed):
            top_z = rng.uniform(0.45, 2.35)
            bm = _build_boulder_bmesh(rng, radius, top_z)
            _create_rock_object(
                name=f"{NAME_PREFIX}Boulder_{boulder_idx + 1:02d}",
                bm=bm,
                location=(bx, by, 0.0),
                col=col,
                mat=mat,
                geo_info=boulder_geo,
                rng=rng,
                rock_tone=rng.uniform(0.05, 0.65),
                has_grass=0.0,
                voxel_size=None,
            )
            placed.append((bx, by, radius))
            boulder_idx += 1

    attempts = 0
    while boulder_idx < boulder_target and attempts < 600:
        attempts += 1
        radius = rng.uniform(2.0, 4.4)
        side_x = rng.uniform(-82.0, -22.0) if (boulder_idx % 2 == 0) else rng.uniform(24.0, 82.0)
        y = rng.uniform(44.0, 158.0)
        if not _is_valid_placement(side_x, y, radius, placed):
            continue

        top_z = rng.uniform(0.35, 2.40)
        bm = _build_boulder_bmesh(rng, radius, top_z)
        _create_rock_object(
            name=f"{NAME_PREFIX}Boulder_{boulder_idx + 1:02d}",
            bm=bm,
            location=(side_x, y, 0.0),
            col=col,
            mat=mat,
            geo_info=boulder_geo,
            rng=rng,
            rock_tone=rng.uniform(0.05, 0.65),
            has_grass=0.0,
            voxel_size=None,
        )
        placed.append((side_x, y, radius))
        boulder_idx += 1

    return col


build()
