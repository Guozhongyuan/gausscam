"""wgpu backend smoke: one synthetic gaussian in front of one camera.

Runs on any Vulkan-capable device (CI: lavapipe; local: NVIDIA). Skipped
when no Vulkan adapter is present. Calibrates nothing at runtime — the
expected center depth pins the camera convention (GL-style: cam_xmat
identity looks along world -z).
"""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

has_vulkan = any(
    a.info["backend_type"] == "Vulkan"
    for a in wgpu.gpu.enumerate_adapters_sync()
)
pytestmark = pytest.mark.gpu
if not has_vulkan:
    pytest.skip("no Vulkan adapter", allow_module_level=True)

from gausscam.backends._wgpu_base import make_K  # noqa: E402
from gausscam.backends.webgpu import Pipeline  # noqa: E402

W = H = 64
DIST = 5.0          # gaussian this far in front of the camera


def _d(cam_pos=(0.0, 0.0, 0.0), cam_xmat=None, n_scene=4):
    cam_xmat = np.eye(3) if cam_xmat is None else cam_xmat
    # robot tail: 3 gaussians of link 0 at the origin; link pose places them
    xyz = np.zeros((3, 3), np.float32)
    # scene head: parked behind the camera so only the robot is visible
    scene = np.full((n_scene, 3), 50.0, np.float32)   # behind cam, out of view
    return {
        "xyz": np.concatenate([scene, xyz]).astype(np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n_scene + 3, 1)),
        "scale": np.full((n_scene + 3, 3), -2.3, np.float32),   # exp ~ 0.1 m
        "opacity": np.full((n_scene + 3,), 10.0, np.float32),   # ~opaque
        "sh": np.zeros((n_scene + 3, 3), np.float32),
        "slots": np.zeros(3, np.int32),                          # 1 link
        "W": np.int32(W), "H": np.int32(H),
        "fovy": np.float32(90.0),
        "cam_pos": np.asarray([cam_pos], np.float32),
        "cam_xmat": np.asarray([cam_xmat], np.float32),
    }


def _pose_at(distance):
    return np.array([[0.0, 0.0, -distance]], np.float32), np.tile(
        np.eye(3, dtype=np.float32)[None], (1, 1))


def test_make_K_vertical_fovy():
    d = _d()
    fxy, cx, cy = make_K(d)
    assert fxy == pytest.approx(H / (2 * np.tan(np.radians(45.0))), rel=1e-5)
    assert cx == W / 2 and cy == H / 2


def test_rejects_non_dc_sh():
    # full INRIA SH block (N, 48) silently misaligns kernel color reads --
    # must be rejected up front, not rendered as fog.
    d = _d()
    d["sh"] = np.zeros((7, 48), np.float32)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        Pipeline(d, "5070" if has_nvidia() else "")


def test_render_shapes_and_dtypes():
    pipe = Pipeline(_d(), "5070" if has_nvidia() else "")
    pf, qf = _pose_at(DIST)
    pipe.set_links(
        np.asarray(pf, np.float32), np.asarray(
            np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(pf), 1)),
            np.float32))
    rgb_p, dep_p = pipe.render_frame()
    rgb, depth = pipe.unpack(rgb_p, dep_p)
    assert rgb.shape == (1, H, W, 3) and rgb.dtype == np.uint8
    assert depth.shape == (1, H, W) and depth.dtype == np.uint16


def test_center_depth_matches_true_distance():
    pipe = Pipeline(_d(), "5070" if has_nvidia() else "")
    pf, _ = _pose_at(DIST)
    pipe.set_links(
        np.asarray(pf, np.float32),
        np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(pf), 1)))
    rgb_p, dep_p = pipe.render_frame()
    _, depth = pipe.unpack(rgb_p, dep_p)
    center = depth[0, H // 2, W // 2]
    assert center == pytest.approx(DIST * 1000, abs=800)   # mm
    # the splat is huge on screen at 5 m: pixel depths are its front shell,
    # so the whole frame brackets the true center distance
    assert depth.min() <= center <= depth.max()
    assert depth.min() > 1000               # nothing at the near plane


def test_render_deterministic():
    d = _d()
    pf, _ = _pose_at(DIST)
    qf = np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(pf), 1))
    a = Pipeline(d, "5070" if has_nvidia() else "")
    a.set_links(np.asarray(pf, np.float32), qf)
    b = Pipeline(d, "5070" if has_nvidia() else "")
    b.set_links(np.asarray(pf, np.float32), qf)
    ra, da = a.render_frame()
    rb, db = b.render_frame()
    assert np.array_equal(ra, rb) and np.array_equal(da, db)


def has_nvidia():
    return any("nvidia" in a.info["device"].lower()
               for a in wgpu.gpu.enumerate_adapters_sync()
               if a.info["backend_type"] == "Vulkan")
