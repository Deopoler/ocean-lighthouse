"""Assemble the full scene in background Blender and save scene.blend.
Usage: blender -b --factory-startup --python scripts/assemble.py -- [--still FRAME OUT.png W H SAMPLES]
"""
import bpy, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = ["ocean_env.py", "lighthouse.py", "rocks.py", "gulls.py", "camera_render.py"]

# empty scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

for name in ORDER:
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print("assemble: SKIP missing", name); continue
    t = time.time()
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), {"__name__": "__main__", "__file__": p})
    print("assemble: ran %s in %.1fs" % (name, time.time() - t))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "scene.blend"))
print("assemble: saved scene.blend")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if argv[:1] == ["--still"]:
    frame, out, w, h, s = int(argv[1]), argv[2], int(argv[3]), int(argv[4]), int(argv[5])
    sc = bpy.context.scene
    sc.frame_set(frame)
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.cycles.samples = s
    sc.cycles.adaptive_threshold = 0.05
    sc.render.filepath = out
    sc.render.image_settings.color_depth = '8'
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print("assemble: still frame %d rendered in %.1fs -> %s" % (frame, time.time() - t, out))
