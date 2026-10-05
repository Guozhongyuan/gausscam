"""Cross-GPU render benchmark on a seeded synthetic scene.

    python -m gausscam.bench [--device-sub SUB] [--n 200000] [--frames 30]
                             [--width 640] [--height 544] [--cams 1]
                             [--seed 42] [--variant auto] [--json PATH]

Inputs are seeded host-side (identical on every device); the report covers
per-frame render time and two structural checks (all frames finite; the
axial landmark's center-pixel depth matches its known distance). Cross-
vendor outputs are NOT compared bitwise. Same device + same seed should
reproduce rgb-sum exactly — useful as a same-card regression anchor.
"""

import argparse
import json
import platform
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(prog="gausscam.bench")
    ap.add_argument("--device-sub", default="")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--cams", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="auto",
                    help="opt | optcol | base | auto (vendor-based)")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args()

    from gausscam.backends.webgpu import Pipeline, pick_variant
    from gausscam.synthetic import make_scene

    variant = (pick_variant(args.device_sub) if args.variant == "auto"
               else args.variant)
    d = make_scene(seed=args.seed, n=args.n, width=args.width,
                   height=args.height, cams=args.cams)
    landmark_mm = float(-d["xyz"][-1, 2] * 1000.0)
    pipe = Pipeline(d, args.device_sub, variant=variant)

    pipe.set_links(np.zeros((1, 3), np.float32),
                   np.array([[1.0, 0, 0, 0]], np.float32))
    pipe.set_cam(d["cam_pos"], d["cam_xmat"])

    for _ in range(3):                                   # warmup
        pipe.render_frame()

    times = []
    rgb_sum = 0
    depth_mm = None
    for _ in range(args.frames):
        t0 = time.perf_counter()
        rgb_p, dep_p = pipe.render_frame()
        times.append((time.perf_counter() - t0) * 1e3)
        rgb, depth_mm = pipe.unpack(rgb_p, dep_p)
        rgb_sum = int(rgb.astype(np.int64).sum())

    finite = bool(np.isfinite(depth_mm).all())
    center = int(depth_mm[0, args.height // 2, args.width // 2])
    ok_landmark = abs(center - landmark_mm) <= 0.05 * landmark_mm

    ads = [a for a in wgpu_adapters()
           if args.device_sub.lower() in a.info["device"].lower()]
    report = dict(
        device=ads[0].info["device"] if ads else "?",
        vendor=ads[0].info["vendor"] if ads else "?",
        variant=variant,
        n=args.n, width=args.width, height=args.height, cams=args.cams,
        frames=args.frames, seed=args.seed,
        ms_mean=round(float(np.mean(times)), 2),
        ms_p50=round(float(np.percentile(times, 50)), 2),
        ms_p95=round(float(np.percentile(times, 95)), 2),
        fps=round(1000.0 / float(np.percentile(times, 50)), 2),
        all_finite=finite, center_depth_mm=center,
        landmark_mm=landmark_mm, landmark_ok=ok_landmark,
        rgb_sum=rgb_sum, host=platform.node(),
    )
    print(json.dumps(report))
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if (finite and ok_landmark) else 1


def wgpu_adapters():
    import wgpu
    return [a for a in wgpu.gpu.enumerate_adapters_sync()
            if a.info["backend_type"] == "Vulkan"]


if __name__ == "__main__":
    raise SystemExit(main())
