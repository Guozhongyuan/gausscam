# gausscam

3D Gaussian Splatting **sensor rendering** for physics simulators. The
physics engine owns the world; gausscam turns a scene splat plus per-link
robot splats into camera-frame sensor images (RGB + metric depth) — with
**no PyTorch** and **one code path for every GPU vendor**.

## Why

We built gausscam after fighting two recurring problems with the obvious
stack (PyTorch + gsplat/CUDA) when rendering is only a *sensor* inside a
simulation loop:

**1. Lightweight.** A simulator host does not need a 2 GB training stack.
gausscam is numpy + `wgpu-py` (Vulkan/Dawn/Metal) only — the whole wheel is
a few hundred KB, `pip install gausscam[webgpu]` pulls no BLAS, no torch,
no CUDA toolkit. CPU-side cost per frame is a sub-millisecond submit plus
two small readbacks; everything else runs on the GPU.

**2. Every GPU vendor, one code path.** The full pipeline (projection,
depth sort, alpha blend) is hand-written WGSL — no cuBLAS/CUB, no vendor
math libraries to break. The same unmodified code runs on NVIDIA
discrete, Intel integrated and (via any conformant Vulkan driver) AMD
GPUs, producing **bit-identical RGB** across vendors (verified: mean
|ΔRGB| = 0.0003/255, max 1/255, on 1.59M gaussians). The known failure
mode this avoids is real: Turing-era cublas `getrsBatched` segfaulted and
corrupted contexts on a driver our first CUDA prototype depended on.
Device quirks still exist (e.g. OneSweep look-back deadlocks Mesa/Intel),
so the pipeline ships two sort variants and picks per-vendor at startup —
but the *contract* never changes.

What gausscam is *not*: a training or graphics tool. There is no
optimization loop, no SSIM loss, no SIBR viewer. Inputs are **activated**
3DGS values (scale in meters, opacity 0..1); outputs are `uint8` RGB and
`uint16` millimeter depth, pixel-aligned with the simulated camera — the
shape a robotics stack actually consumes.

## Layout

```
gausscam/
├── core/       simulator-agnostic, GPU-agnostic
│   ├── assets.py    GaussianCloud / SceneSplat / RobotSplat (+ merge)
│   ├── rig.py       SensorRig / SensorCam (extrinsics in a link frame)
│   ├── backend.py   RenderBackend protocol -> Frame(rgb u8, depth u16 mm)
│   └── cull.py      SelfCull (hide the robot's own splats near a sensor)
├── backends/   the GPU axis
│   └── webgpu.py    standalone wgpu pipeline (wgpu-py 0.32 → Vulkan;
│                     OneSweep sort on NVIDIA, hierarchical scan
│                     elsewhere; ~15 ms stereo @ 1.6M gaussians on an
│                     RTX 5070). _wgpu_base.py carries the WGSL kernels.
└── adapters/   the simulator axis (one pose contract)
    └── mujoco.py    mj_data.xpos/xquat (native wxyz) + cam_xpos/cam_xmat
    ( future: mjx batch / motrixsim / gazebo )
```

Pose contract: `{link_name: (pos[3] f32, quat_wxyz[4] f32)}`.
Per frame: `sim step → adapter.body_poses(...) → backend.set_links(...) →
backend.render(...) → numpy frames`.

Install: `pip install gausscam` (core, numpy-only); add `[webgpu]` for the
Vulkan render pipeline, `[mujoco]` for the adapter, `[io]` for 3DGS PLY
loading.

## Example 1: render any 3DGS PLY (no simulator)

`pip install "gausscam[webgpu,io,dev]"`, pick a viewpoint, render RGB + mm
depth to disk. Works with any INRIA-layout `ply` (the standard output of
3DGS training tools):

```python
import numpy as np
from PIL import Image
from gausscam.core.assets import GaussianCloud
from gausscam.backends.webgpu import Pipeline, pick_variant

CLOUD_PLY = "splat.ply"
EYE = np.array([-7.0, 1.0, 11.0], np.float32)     # camera position (world)
TARGET = np.array([-7.5, 4.0, 5.0], np.float32)   # look at this point
FOVY, W, H = 90.0, 640, 544

g = GaussianCloud.from_ply(CLOUD_PLY)             # RAW INRIA values

# activate: kernels consume scale in METERS and opacity in 0..1; the DC
# color term feeds as-is (color = 0.2821 * sh + 0.5, degree 0 only)
d = dict(
    xyz=np.ascontiguousarray(g.xyz, np.float32),
    rot=g.rot / np.linalg.norm(g.rot, axis=1, keepdims=True),
    scale=np.exp(g.scale).astype(np.float32),
    opacity=(1.0 / (1.0 + np.exp(-g.opacity))).astype(np.float32),
    sh=g.sh[:, :3],
    slots=np.zeros(len(g.xyz), np.int32),   # whole cloud rides link slot 0
    W=np.int32(W), H=np.int32(H), fovy=np.float32(FOVY),
)

# look-at camera (OpenGL/MuJoCo convention: the camera looks along its -z)
fwd = TARGET - EYE; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
d["cam_pos"] = EYE[None]
d["cam_xmat"] = np.column_stack([right, up, -fwd]).astype(np.float32)[None]

sub = "5070"                                    # substring of your GPU name
pipe = Pipeline(d, sub, variant=pick_variant(sub))
pipe.set_links(np.zeros((1, 3), np.float32),    # park slot 0 at identity
               np.array([[1.0, 0, 0, 0]], np.float32))
rgb, depth = pipe.unpack(*pipe.render_frame())
Image.fromarray(rgb[0]).save("render.png")
print("depth mm: p50", np.median(depth[0]), "max", depth[0].max())
```

`pick_variant` keeps one code path per vendor (see Known device quirks in
[docs/benchmarks.md](docs/benchmarks.md)); pass any substring of your
adapter name as `sub`. Viewpoints outside the capture volume show the
usual 3DGS fog — gausscam renders whatever the splats encode.

Rendered with this snippet (983k-gaussian stairwell scan):

![stairwell rendered from a 3DGS PLY](https://raw.githubusercontent.com/Guozhongyuan/gausscam/main/docs/images/ply-example.jpg)

## Example 2: MuJoCo — a stepping box splat, 30 lines

`pip install "gausscam[webgpu,mujoco]"`, then — the simulator owns the
world, gausscam renders its sensor view of the splat:

```python
import mujoco
import numpy as np

from gausscam.adapters.mujoco import MuJoCoAdapter
from gausscam.backends.webgpu import Pipeline

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <camera name="cam" pos="0 -1.2 0" xyaxes="1 0 0 0 0 1"/>
    <body name="box" pos="0 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>""")
data = mujoco.MjData(model)

# three gaussians as the box's splat (slot 0 = its only link; no scene here)
N = 3
d = dict(
    xyz=np.zeros((N, 3), np.float32),
    rot=np.tile([1.0, 0, 0, 0], (N, 1)).astype(np.float32),
    scale=np.full((N, 3), 0.05, np.float32),      # meters (activated)
    opacity=np.full(N, 1.0, np.float32),          # 0..1 (activated)
    sh=np.zeros((N, 3), np.float32),
    slots=np.arange(N, dtype=np.int32),
    W=np.int32(320), H=np.int32(240), fovy=np.float32(90.0),
    cam_pos=np.zeros((1, 3), np.float32),
    cam_xmat=np.eye(3, dtype=np.float32)[None],
)
ad, pipe = MuJoCoAdapter(model, data), Pipeline(d)       # first Vulkan device

data.qvel[0] = 0.5                          # slide the box along +x
for _ in range(120):
    mujoco.mj_step(model, data)
    pos, quat = ad.body_poses(["box"])["box"]            # <- the pose contract
    pipe.set_links(pos[None], quat[None])
    pipe.set_cam(*ad.camera_poses(["cam"]))
    rgb, depth = pipe.unpack(*pipe.render_frame())

print(rgb.shape, depth.shape)   # (1, 240, 320, 3) uint8  (1, 240, 320) uint16 mm
```

The 3DGS dict fields are **activated** values — what the kernels consume:
`scale` in meters, `opacity` in 0..1. (`GaussianCloud.from_ply` returns
the raw INRIA PLY values, which are log-scale/logit; apply `exp`/`sigmoid`
before feeding a Pipeline.) `slots[i]` assigns robot gaussian i to a link
slot; `set_links` poses are per slot in `RobotSplat.link_names` order.

For real scenes: load a captured scene with `SceneSplat.load(ply)` and
robot links with `RobotSplat.load_dir()` (needs `[io]`), merge with
`gausscam.core.assets.merge`, and feed the merged dict the same way.

## Benchmarks

`python -m gausscam.bench` renders a seeded synthetic scene (default
200k gaussians, stereo 640x544) and checks geometric correctness; use it
to verify a new GPU/driver before relying on it. Cross-vendor numbers and
the bit-identical-output verification are in
[docs/benchmarks.md](docs/benchmarks.md).

## Licenses

- gausscam: MIT (see `LICENSE`).
- The WGSL kernels were clean-room written against the public 3DGS
  equations; no INRIA/graphdeco code is included.
