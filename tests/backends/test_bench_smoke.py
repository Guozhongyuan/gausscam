"""Bench module smoke: tiny seeded scene, run in a SUBPROCESS.

Subprocess because (a) a wgpu/Mesa panic on some adapters aborts the whole
pytest process, and (b) the project's known exit-time GL crash family --
isolating both from the suite. Skipped when no Vulkan adapter exists.
"""

import subprocess
import sys

import pytest

wgpu = pytest.importorskip("wgpu")

has_vulkan = any(
    a.info["backend_type"] == "Vulkan"
    for a in wgpu.gpu.enumerate_adapters_sync()
)
pytestmark = pytest.mark.gpu
if not has_vulkan:
    pytest.skip("no Vulkan adapter", allow_module_level=True)

LANDMARK_MM = 4000.0

RUNNER = r"""
import numpy as np
from gausscam.synthetic import make_scene
from gausscam.backends.webgpu import Pipeline, pick_variant

W, H = 640, 544
LANDMARK_MM = 4000.0
# n=200k: Intel ARL ANV aborts on small dispatches (n<=20k crashes, 200k is
# fine) -- use the bench-proven config so one scene works on every vendor.
d = make_scene(seed=42, n=200000, width=W, height=H, cams=1,
               landmark_mm=LANDMARK_MM)
# explicit adapter: host enumeration order varies and the software adapters
# panic on tiny dispatches -- prefer real hardware, else skip loudly
import wgpu
soft = ("llvmpipe", "lavapipe", "swiftshader")
real = [a for a in wgpu.gpu.enumerate_adapters_sync()
        if a.info["backend_type"] == "Vulkan"
        and not any(s in a.info["device"].lower() for s in soft)]
if not real:
    print("NO-HW")
    raise SystemExit(3)
sub = real[0].info["device"][:20]
pipe = Pipeline(d, sub, variant=pick_variant(sub))   # opt deadlocks Mesa/Intel
pipe.set_links(np.zeros((1, 3), np.float32),
               np.array([[1.0, 0, 0, 0]], np.float32))
pipe.set_cam(d["cam_pos"], d["cam_xmat"])
for _ in range(2):
    pipe.render_frame()
rgb, depth = pipe.unpack(*pipe.render_frame())
assert np.isfinite(depth).all()
center = int(depth[0, H // 2, W // 2])
print(f"CENTER={center}", flush=True)
import os
os._exit(0)                     # dodge the known exit-time wgpu/EGL crash
"""


def test_bench_scene_geometry():
    r = subprocess.run([sys.executable, "-c", RUNNER], capture_output=True,
                       text=True, timeout=180)
    if "NO-HW" in r.stdout:
        pytest.skip("no hardware Vulkan adapter")
    assert r.returncode == 0, f"runner failed rc={r.returncode}\n{r.stderr[-800:]}"
    center = int(r.stdout.strip().split("CENTER=")[1])
    assert center == pytest.approx(LANDMARK_MM, rel=0.02), (
        f"axial landmark depth {center} != {LANDMARK_MM}"
    )
