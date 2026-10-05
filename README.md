# gausscam

3D Gaussian Splatting **sensor rendering** for physics simulators. The
physics engine owns the world; gausscam turns a scene splat plus per-link
robot splats into camera-frame sensor images (RGB + metric depth).

## Why

Simulation hosts don't need a training stack, and sensor loops shouldn't
break on vendor math libraries:

- **Lightweight** — numpy + `wgpu-py` only. No torch, no CUDA toolkit;
  the wheel is a few hundred KB and host cost per frame is a
  sub-millisecond submit plus two small readbacks.
- **Every GPU vendor, one code path** — projection, sorting and blending
  are hand-written WGSL (no cuBLAS/CUB). The same code runs on NVIDIA
  discrete, Intel integrated and AMD (any conformant Vulkan driver),
  bit-identical RGB across vendors (max diff 1/255 on 1.59M gaussians).
  Device quirks are handled by shipping two sort variants and picking
  per-vendor at startup.

Not a training or graphics tool: inputs are **activated** 3DGS values,
outputs are `uint8` RGB + `uint16` millimeter depth, pixel-aligned with
the simulated camera.

## Layout

```
gausscam/
├── core/       simulator-agnostic, GPU-agnostic
│   ├── assets.py    GaussianCloud / SceneSplat / RobotSplat (+ merge)
│   ├── rig.py       SensorRig / SensorCam (extrinsics in a link frame)
│   ├── backend.py   RenderBackend protocol -> Frame(rgb u8, depth u16 mm)
│   └── cull.py      SelfCull (hide the robot's own splats near a sensor)
├── backends/   the GPU axis
│   └── webgpu.py    wgpu pipeline (Vulkan; ~15 ms stereo @ 1.6M
│                     gaussians on an RTX 5070). _wgpu_base.py has the
│                     WGSL kernels.
└── adapters/   the simulator axis (one pose contract)
    └── mujoco.py    mj_data.xpos/xquat (native wxyz) + cam_xpos/cam_xmat
    ( future: mjx batch / motrixsim / gazebo )
```

Pose contract: `{link_name: (pos[3] f32, quat_wxyz[4] f32)}`. Per frame:
`sim step → adapter.body_poses(...) → backend.set_links(...) →
backend.render(...) → numpy frames`.

Install: `pip install gausscam` (core); add `[webgpu]` for the render
pipeline, `[mujoco]` for the adapter, `[io]` for PLY loading.

## Example 1: render any 3DGS PLY (no simulator)

Standard INRIA and super-splat compressed PLYs are auto-detected from the
vertex element. `pip install "gausscam[webgpu,io,dev]"`, pick a viewpoint,
render:

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
    slots=np.zeros(0, np.int32),            # static-only cloud: no robot block
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
rgb, depth = pipe.unpack(*pipe.render_frame())  # static: no set_links needed
Image.fromarray(rgb[0]).save("render.png")
```

Output (7.7M-gaussian church scan rendered at the simulation spawn with
the full cloud; viewpoints outside the capture volume show the usual
3DGS fog):

![church nave rendered from a 3DGS PLY](https://raw.githubusercontent.com/Guozhongyuan/gausscam/main/docs/images/church-example.jpg)

## Example 2: MuJoCo

`pip install "gausscam[webgpu,mujoco]"` — the simulator owns the world,
gausscam renders its sensor view of the splat:

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

# three gaussians as the box's splat (slot 0 = its only link)
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
ad, pipe = MuJoCoAdapter(model, data), Pipeline(d)   # first Vulkan device

data.qvel[0] = 0.5                          # slide the box along +x
for _ in range(120):
    mujoco.mj_step(model, data)
    pos, quat = ad.body_poses(["box"])["box"]            # <- the pose contract
    pipe.set_links(pos[None], quat[None])
    pipe.set_cam(*ad.camera_poses(["cam"]))
    rgb, depth = pipe.unpack(*pipe.render_frame())

print(rgb.shape, depth.shape)   # (1, 240, 320, 3) uint8  (1, 240, 320) uint16 mm
```

For real scenes: `SceneSplat.load(ply)` + `RobotSplat.load_dir()` (needs
`[io]`), merge with `gausscam.core.assets.merge`, feed the merged dict
the same way (scene without a robot: empty `slots`, as above).

## Benchmarks

`python -m gausscam.bench` renders a seeded synthetic scene and checks
geometric correctness — run it to verify a new GPU/driver. Cross-vendor
numbers: [docs/benchmarks.md](docs/benchmarks.md).

## License

MIT (see `LICENSE`). The WGSL kernels are clean-room written against the
public 3DGS equations; no INRIA/graphdeco code is included.
