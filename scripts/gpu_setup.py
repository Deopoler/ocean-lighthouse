import bpy
p = bpy.context.preferences.addons['cycles'].preferences
p.compute_device_type = 'ONEAPI'
p.refresh_devices()
if hasattr(p, 'use_oneapirt'):
    p.use_oneapirt = False  # Embree GPU (hardware RT) fails on Arc 130V
for d in p.devices:
    d.use = (d.type == 'ONEAPI')
for s in bpy.data.scenes:
    s.cycles.device = 'GPU'
    s.render.use_overwrite = False  # skip frames already rendered (resume)
    s.render.use_placeholder = False
print("GPU_SETUP", [(d.name, d.use) for d in p.devices])
