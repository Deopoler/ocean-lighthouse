# Ocean Lighthouse (Blender procedural animation)

A 10-second (240 frames, 24 fps, 1920x1080) Cycles animation of a lighthouse on a rocky islet at golden hour, built entirely from Python scripts: procedural ocean, sea stacks, lighthouse with a rotating beam, and a flock of animated gulls.

Result: [`render/ocean_lighthouse.mp4`](render/ocean_lighthouse.mp4)

## Scripts

| Script | What it builds |
|---|---|
| `scripts/ocean_env.py` | Ocean modifier surface, sky, sun |
| `scripts/rocks.py` | Fractured sea stacks and boulders |
| `scripts/lighthouse.py` | Lighthouse, keeper's house, rotating beam |
| `scripts/gulls.py` | Rigged gulls with flapping animation |
| `scripts/camera_render.py` | Camera path, depth of field, render settings |
| `scripts/assemble.py` | Runs all of the above and saves `scene.blend` |
| `scripts/render_frames.ps1` | Renders frames and encodes the MP4 with ffmpeg |
| `scripts/gpu_setup.py` | Switches Cycles to the Intel Arc GPU (oneAPI) |

## Build and render (Blender 5.3)

```powershell
blender -b --factory-startup --python scripts/assemble.py
./scripts/render_frames.ps1            # CPU (about 1 min/frame)
./scripts/render_frames.ps1 -Gpu       # Intel Arc GPU, leaves the CPU mostly free; skips existing frames
```

On Intel Arc, Cycles GPU rendering fails with an Embree BVH error unless oneAPI hardware ray tracing is off; `gpu_setup.py` turns it off.
