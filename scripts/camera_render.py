"""Configure the animated drone camera and final Cycles render settings."""

import ast
import math
import re
import bpy


def _get_or_create_object(name, data=None, scene=None):
    """Return an existing object by name (linked to scene) or create and link it."""
    if scene is None:
        scene = bpy.context.scene
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
    elif obj.name not in scene.objects:
        scene.collection.objects.link(obj)
    return obj


def _get_fcurves(obj):
    """Return Blender 5.x layered-action F-curves for an animated object."""
    anim = obj.animation_data
    if anim is None or anim.action is None:
        return []
    action = anim.action
    layer = action.layers[0]
    strip = layer.strips[0]
    slot = action.slots[0]
    return strip.channelbag(slot).fcurves


def _set_ease_in_out(obj):
    """Set BEZIER + AUTO_CLAMPED handles on all keyframes of obj for smooth ease-in/out."""
    for fcurve in _get_fcurves(obj):
        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"
        fcurve.update()


def _get_enum_items(struct, prop_name):
    """Read enum identifiers at runtime, including dynamic enums."""
    prop = struct.bl_rna.properties[prop_name]
    items = [item.identifier for item in prop.enum_items]
    if items and items != ["NONE"]:
        return items

    # Dynamic enums (such as color management look/view_transform) expose valid
    # identifiers in the TypeError raised on an invalid assignment.
    current = getattr(struct, prop_name)
    try:
        setattr(struct, prop_name, "__ENUM_PROBE__")
    except TypeError as exc:
        match = re.search(r"not found in (\(.*\))", str(exc))
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
                if isinstance(parsed, tuple):
                    return [str(v) for v in parsed]
            except (SyntaxError, ValueError):
                pass
    else:
        setattr(struct, prop_name, current)

    return items


def _configure_camera(scene):
    """Set up Camera, CAM_Rig, CAM_Target, animation, banking, shake, and lens/DoF."""
    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj is None or cam_obj.type != "CAMERA":
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam_obj)
    else:
        cam_data = cam_obj.data
        if cam_obj.name not in scene.objects:
            scene.collection.objects.link(cam_obj)

    cam_target = _get_or_create_object("CAM_Target", data=None, scene=scene)
    cam_target.empty_display_type = "PLAIN_AXES"
    cam_target.empty_display_size = 2.0

    cam_rig = _get_or_create_object("CAM_Rig", data=None, scene=scene)
    cam_rig.empty_display_type = "ARROWS"
    cam_rig.empty_display_size = 2.0

    # Clear existing animation data and Track To constraints for idempotency.
    for obj in (cam_obj, cam_rig, cam_target):
        obj.animation_data_clear()
    for con in list(cam_obj.constraints):
        if con.type == "TRACK_TO":
            cam_obj.constraints.remove(con)
    for con in list(cam_rig.constraints):
        if con.type == "TRACK_TO":
            cam_rig.constraints.remove(con)

    # Parent Camera to CAM_Rig so Track To on CAM_Rig does not override local roll/shake.
    cam_obj.parent = cam_rig
    cam_obj.matrix_parent_inverse.identity()
    cam_obj.location = (0.0, 0.0, 0.0)
    cam_obj.rotation_mode = "XYZ"
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)

    # Flight path on CAM_Rig: low drone skimming the waves with a gentle curve and rise.
    rig_keyframes = [
        (1, (-20.0, -80.0, 3.2)),
        (120, (-17.5, -22.0, 4.2)),
        (240, (-6.0, 38.0, 6.0)),
    ]
    for frame, loc in rig_keyframes:
        cam_rig.location = loc
        cam_rig.keyframe_insert(data_path="location", frame=frame)
    _set_ease_in_out(cam_rig)

    # Target drift across the lighthouse islet.
    target_keyframes = [
        (1, (5.0, 80.0, 8.0)),
        (240, (10.0, 80.0, 14.0)),
    ]
    for frame, loc in target_keyframes:
        cam_target.location = loc
        cam_target.keyframe_insert(data_path="location", frame=frame)
    _set_ease_in_out(cam_target)

    # Track To constraint on CAM_Rig aiming at CAM_Target.
    track = cam_rig.constraints.new(type="TRACK_TO")
    track.name = "Track To"
    track.target = cam_target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Subtle roll (bank) along the curve + baseline location key on Camera for noise modifiers.
    cam_obj.location = (0.0, 0.0, 0.0)
    cam_obj.keyframe_insert(data_path="location", frame=1)

    bank_keyframes = [
        (1, (0.0, 0.0, 0.0)),
        (120, (0.0, 0.0, math.radians(1.8))),
        (240, (0.0, 0.0, 0.0)),
    ]
    for frame, rot in bank_keyframes:
        cam_obj.rotation_euler = rot
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    _set_ease_in_out(cam_obj)

    # Subtle low-frequency handheld/drone shake via Noise F-curve modifiers.
    loc_noise_params = [
        (32.0, 0.08, 0.0),
        (38.0, 0.08, 17.3),
        (26.0, 0.08, 41.7),
    ]
    rot_noise_params = [
        (28.0, 0.012, 73.1),
        (35.0, 0.012, 109.5),
        (40.0, 0.012, 151.9),
    ]
    for fcurve in _get_fcurves(cam_obj):
        if fcurve.data_path == "location":
            scale, strength, phase = loc_noise_params[fcurve.array_index]
        elif fcurve.data_path == "rotation_euler":
            scale, strength, phase = rot_noise_params[fcurve.array_index]
        else:
            continue
        mod = fcurve.modifiers.new(type="NOISE")
        mod.scale = scale
        mod.strength = strength
        mod.phase = phase
        mod.depth = 2

    # Lens, sensor, clipping, and subtle depth of field focused on CAM_Target.
    cam_data.lens = 28.0
    cam_data.sensor_width = 36.0
    cam_data.clip_end = 20000.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = cam_target
    cam_data.dof.aperture_fstop = 8.0

    scene.camera = cam_obj
    return cam_obj


def _configure_cycles_device(scene):
    """Enable ONEAPI GPU devices only when available; otherwise fall back to CPU."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    gpu_enabled = False
    try:
        prefs.compute_device_type = "ONEAPI"
        prefs.refresh_devices()
        oneapi_devices = [d for d in prefs.devices if d.type == "ONEAPI"]
        if oneapi_devices:
            for dev in prefs.devices:
                dev.use = dev.type == "ONEAPI"
            gpu_enabled = prefs.has_active_device()
    except Exception:
        gpu_enabled = False

    if gpu_enabled:
        scene.cycles.device = "GPU"
    else:
        try:
            prefs.compute_device_type = "NONE"
        except Exception:
            pass
        scene.cycles.device = "CPU"


def _configure_render_settings(scene):
    """Configure Cycles, resolution, timing, sampling, denoiser, light paths, color, and output."""
    render = scene.render
    cycles = scene.cycles

    render.engine = "CYCLES"
    _configure_cycles_device(scene)

    render.resolution_x = 1920
    render.resolution_y = 1080
    render.resolution_percentage = 100

    scene.frame_start = 1
    scene.frame_end = 240
    render.fps = 24
    render.fps_base = 1.0

    cycles.use_adaptive_sampling = True
    cycles.samples = 96
    cycles.adaptive_threshold = 0.02

    cycles.use_denoising = True
    denoiser_items = _get_enum_items(cycles, "denoiser")
    if "OPENIMAGEDENOISE" in denoiser_items:
        cycles.denoiser = "OPENIMAGEDENOISE"

    cycles.max_bounces = 6
    cycles.transparent_max_bounces = 12
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.blur_glossy = 1.0
    cycles.sample_clamp_indirect = 8.0

    render.use_motion_blur = True
    render.motion_blur_shutter = 0.5

    view_settings = scene.view_settings
    vt_items = _get_enum_items(view_settings, "view_transform")
    agx_match = next((item for item in vt_items if item.lower() == "agx"), None)
    if agx_match:
        view_settings.view_transform = agx_match
        look_items = _get_enum_items(view_settings, "look")
        look_match = next(
            (item for item in look_items if "medium high contrast" in item.lower()),
            None,
        )
        if look_match:
            view_settings.look = look_match

    render.use_persistent_data = True
    render.film_transparent = False

    render.filepath = "//render/frames/frame_####.png"
    render.use_file_extension = True
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.image_settings.color_depth = "16"

    render.use_overwrite = False
    render.use_placeholder = True


def setup_camera_and_render(scene=None):
    """Apply camera animation and final Cycles render settings to the given scene."""
    if scene is None:
        scene = bpy.context.scene
    cam_obj = _configure_camera(scene)
    _configure_render_settings(scene)
    return cam_obj


def main():
    """Run camera and render setup and print camera world positions at key frames."""
    scene = bpy.context.scene
    cam_obj = setup_camera_and_render(scene)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for frame in (1, 120, 240):
        scene.frame_set(frame)
        depsgraph.update()
        pos = cam_obj.matrix_world.translation
        print(f"Frame {frame:3d}: Camera world position = ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")


if __name__ == "__main__":
    main()
