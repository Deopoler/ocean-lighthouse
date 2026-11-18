"""Ocean, distant sea, and golden-hour sky. Idempotent; uses collection 'Environment'."""
import bpy, bmesh, math

COLL = "Environment"


def _coll():
    c = bpy.data.collections.get(COLL)
    if c:
        for o in list(c.objects):
            bpy.data.objects.remove(o, do_unlink=True)
    else:
        c = bpy.data.collections.new(COLL)
        bpy.context.scene.collection.children.link(c)
    return c


def _water_material():
    mat = bpy.data.materials.get("Water") or bpy.data.materials.new("Water")
    mat.use_nodes = True
    nt = mat.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new('ShaderNodeOutputMaterial'); out.location = (900, 0)
    water = N.new('ShaderNodeBsdfPrincipled'); water.location = (300, 200)
    water.inputs['Roughness'].default_value = 0.025
    water.inputs['IOR'].default_value = 1.333
    geo = N.new('ShaderNodeNewGeometry')
    sep = N.new('ShaderNodeSeparateXYZ'); L.new(geo.outputs['Position'], sep.inputs[0])
    mr = N.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = 0.3; mr.inputs['From Max'].default_value = 2.2
    L.new(sep.outputs['Z'], mr.inputs['Value'])
    mix = N.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (0.004, 0.018, 0.025, 1)
    mix.inputs['B'].default_value = (0.02, 0.11, 0.10, 1)
    L.new(mr.outputs[0], mix.inputs['Factor'])
    L.new(mix.outputs['Result'], water.inputs['Base Color'])
    attr = N.new('ShaderNodeAttribute'); attr.attribute_name = 'foam'
    noise = N.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 0.35; noise.inputs['Detail'].default_value = 12
    noise.inputs['Roughness'].default_value = 0.65
    mul = N.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
    L.new(attr.outputs['Fac'], mul.inputs[0]); L.new(noise.outputs['Fac'], mul.inputs[1])
    ramp = N.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.18; ramp.color_ramp.elements[1].position = 0.42
    L.new(mul.outputs[0], ramp.inputs['Fac'])
    foam = N.new('ShaderNodeBsdfPrincipled')
    foam.inputs['Base Color'].default_value = (0.75, 0.78, 0.78, 1)
    foam.inputs['Roughness'].default_value = 0.6
    foam.inputs['Subsurface Weight'].default_value = 0.3
    ms = N.new('ShaderNodeMixShader')
    L.new(ramp.outputs['Color'], ms.inputs['Fac'])
    L.new(water.outputs[0], ms.inputs[1]); L.new(foam.outputs[0], ms.inputs[2])
    L.new(ms.outputs[0], out.inputs['Surface'])
    tc = N.new('ShaderNodeTexCoord')
    n2 = N.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 0.9; n2.inputs['Detail'].default_value = 12; n2.inputs['Roughness'].default_value = 0.62
    L.new(tc.outputs['Object'], n2.inputs['Vector'])
    bump = N.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.3; bump.inputs['Distance'].default_value = 0.08
    L.new(n2.outputs['Fac'], bump.inputs['Height'])
    L.new(bump.outputs['Normal'], water.inputs['Normal'])
    return mat


def _far_material():
    fm = bpy.data.materials.get("FarWater") or bpy.data.materials.new("FarWater")
    fm.use_nodes = True
    nt = fm.node_tree
    for n in list(nt.nodes):
        if n.type not in ('BSDF_PRINCIPLED', 'OUTPUT_MATERIAL'):
            nt.nodes.remove(n)
    p = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    p.inputs['Base Color'].default_value = (0.004, 0.018, 0.025, 1)
    p.inputs['Roughness'].default_value = 0.07
    fn = nt.nodes.new('ShaderNodeTexNoise'); fn.inputs['Scale'].default_value = 0.02; fn.inputs['Detail'].default_value = 14
    fb = nt.nodes.new('ShaderNodeBump'); fb.inputs['Strength'].default_value = 0.25
    nt.links.new(fn.outputs['Fac'], fb.inputs['Height']); nt.links.new(fb.outputs['Normal'], p.inputs['Normal'])
    return fm


def build():
    sc = bpy.context.scene
    c = _coll()
    # Ocean
    me = bpy.data.meshes.new("OceanMesh")
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0); bm.to_mesh(me); bm.free()
    oc = bpy.data.objects.new("Ocean", me); c.objects.link(oc)
    oc.location = (0, 60, 0)
    m = oc.modifiers.new('Ocean', 'OCEAN')
    m.geometry_mode = 'GENERATE'
    m.size = 1.0; m.spatial_size = 50; m.repeat_x = 12; m.repeat_y = 12
    m.resolution = 11; m.viewport_resolution = 6  # faces per tile side = res^2
    m.wave_scale = 2.0; m.choppiness = 1.6; m.wind_velocity = 22
    m.wave_alignment = 0.45; m.wave_direction = 0.4; m.damping = 0.5
    m.use_normals = True
    m.use_foam = True; m.foam_layer_name = 'foam'; m.foam_coverage = 0.35
    try:
        m.spectrum = 'JONSWAP'
    except Exception:
        pass
    # linear time via driver: time = 1 + frame/24*0.9
    d = m.driver_add('time').driver
    d.type = 'SCRIPTED'; d.expression = '1 + frame/24*0.9'
    # secondary wind chop on top of the swell
    tex = bpy.data.textures.get("OceanChop") or bpy.data.textures.new("OceanChop", 'CLOUDS')
    tex.noise_scale = 3.5; tex.noise_depth = 2; tex.noise_basis = 'ORIGINAL_PERLIN'
    dm = oc.modifiers.new('Chop', 'DISPLACE'); dm.texture = tex; dm.strength = 0.25; dm.mid_level = 0.5
    dm.texture_coords = 'GLOBAL'
    for p in me.polygons:
        p.use_smooth = True
    # GENERATE mode does not center the tiled mesh on the object origin; re-center it on (0, 60).
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    ev = oc.evaluated_get(dg)
    xs = [v.co.x for v in ev.data.vertices]; ys = [v.co.y for v in ev.data.vertices]
    oc.location = (-(min(xs) + max(xs)) / 2, 60 - (min(ys) + max(ys)) / 2, 0)
    print("ocean_env: local bounds x %.1f..%.1f y %.1f..%.1f" % (min(xs), max(xs), min(ys), max(ys)))
    me.materials.append(_water_material())
    # Distant sea ring (hole where the ocean mesh is)
    fme = bpy.data.meshes.new("FarSeaMesh")
    bm = bmesh.new()
    outer, inner = 10000, 290
    vo = [bm.verts.new((x, y, 0)) for x, y in [(-outer, -outer), (outer, -outer), (outer, outer), (-outer, outer)]]
    vi = [bm.verts.new((x, y, 0)) for x, y in [(-inner, -inner), (inner, -inner), (inner, inner), (-inner, inner)]]
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((vo[i], vo[j], vi[j], vi[i]))
    bm.to_mesh(fme); bm.free()
    far = bpy.data.objects.new("FarSea", fme); c.objects.link(far)
    far.location = (0, 60, 0)
    fme.materials.append(_far_material())
    # Sky
    w = sc.world or bpy.data.worlds.new('World'); sc.world = w
    w.use_nodes = True; nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    sky = nt.nodes.new('ShaderNodeTexSky')
    for t in ('MULTIPLE_SCATTERING', 'SINGLE_SCATTERING', 'NISHITA'):
        try:
            sky.sky_type = t; break
        except TypeError:
            pass
    sky.sun_disc = True; sky.sun_size = math.radians(0.6); sky.sun_intensity = 1.0
    sky.sun_elevation = math.radians(3.0); sky.sun_rotation = math.radians(0)
    sky.altitude = 50; sky.air_density = 1.2; sky.aerosol_density = 2.5; sky.ozone_density = 1.0
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = 0.22
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(sky.outputs['Color'], bg.inputs['Color']); nt.links.new(bg.outputs[0], out.inputs['Surface'])
    sc.frame_start, sc.frame_end = 1, 240
    sc.render.fps = 24
    print("ocean_env: built")


build()
