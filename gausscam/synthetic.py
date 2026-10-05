"""Deterministic synthetic scenes for cross-GPU benchmarks.

Everything is seeded host-side (numpy default_rng) so the INPUT is
bit-identical on every device; outputs are not compared bitwise across
vendors (float order differs) — bench.py checks structural invariants
instead. The gaussian cloud is all in link slot 0 with an identity link
pose, so xyz renders as-is.

Layout mimics REAL capture statistics — many SMALL splats spread over a
wall (1-2 tiles each on screen; the real map3 data renders 1.6M in ~30 ms
precisely because its splats are tiny on screen). Piling big overlapping
blobs instead hits the pipeline's (gaussian, tile) pair budget (~4M) and
deadlocks it — that mode is available as layout="stress" for driver
robustness testing, NOT for timing.

An "axial landmark" gaussian floats alone on the camera's center ray in
front of the wall; bench.py checks the center pixel's depth against its
known distance as the analytic anchor.
"""

import numpy as np


def make_scene(
    seed: int = 42,
    n: int = 200_000,
    width: int = 640,
    height: int = 544,
    fovy: float = 90.0,
    cams: int = 1,
    layout: str = "wall",
    landmark_mm: float = 4000.0,
    wall_z: float = -6.0,
) -> dict:
    """Full Pipeline-ready dict. n includes the axial landmark gaussian."""
    if n < 2:
        raise ValueError("n must be >= 2 (cloud + axial landmark)")
    if layout not in ("wall", "stress"):
        raise ValueError(f"unknown layout {layout!r}")

    rng = np.random.default_rng(seed)
    n_cloud = n - 1

    # Wall of small splats filling the frustum at wall_z: half-extents cover
    # the fovy/2 cone (tan(45deg)*|z|) with margin, so most tiles get work.
    half_h = float(np.tan(np.radians(fovy / 2)) * abs(wall_z))
    half_w = half_h * width / height
    xyz = np.zeros((n_cloud, 3), np.float32)
    xyz[:, 0] = rng.uniform(-half_w, half_w, n_cloud)
    xyz[:, 1] = rng.uniform(-half_h, half_h, n_cloud)
    xyz[:, 2] = wall_z + rng.normal(0, 0.05, n_cloud)      # shallow depth fold
    # ACTIVATED values, matching what Pipeline kernels consume (the real
    # w1_static.npz stores scale in meters (mean ~2 cm) and opacity in 0..1):
    # pixel-scale splats look exactly like real captures on screen.
    scales = rng.uniform(0.005, 0.04, (n_cloud, 3))         # meters
    opacities = rng.uniform(0.5, 1.0, n_cloud)              # 0..1

    # axial landmark: alone on the center ray, 2 m in front of the wall
    landmark = np.array([[0.0, 0.0, -landmark_mm / 1000.0]], np.float32)
    landmark_scale = np.full((1, 3), -3.0, np.float32)      # ~5 cm sigma

    if layout == "stress":
        # near-CAP overdraw: pull the cloud close and inflate the splats.
        # Deadlocks the pipeline past ~100k gaussians -- driver robustness
        # probe only, never a timing workload.
        xyz[:, :2] *= 0.3
        xyz[:, 2] = wall_z / 3 + rng.normal(0, 0.3, n_cloud)
        scales = np.full((n_cloud, 3), 0.30, np.float32)

    return dict(
        xyz=np.concatenate([xyz, landmark]).astype(np.float32),
        rot=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1)),
        scale=np.concatenate([scales, landmark_scale]).astype(np.float32),
        opacity=np.concatenate([opacities, [1.0]]).astype(np.float32),
        sh=np.zeros((n, 3), np.float32),
        slots=np.zeros(n, np.int32),                        # all in link 0
        W=np.int32(width),
        H=np.int32(height),
        fovy=np.float32(fovy),
        cam_pos=np.zeros((cams, 3), np.float32),
        cam_xmat=np.tile(np.eye(3, dtype=np.float32)[None], (cams, 1, 1)),
    )
