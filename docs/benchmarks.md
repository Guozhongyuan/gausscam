# gausscam benchmarks

All numbers from the standalone wgpu backend (`gausscam.backends.webgpu`)
on unmodified public drivers. Reproduce with:

```bash
pip install "gausscam[webgpu,dev]"
python -m gausscam.bench --device-sub <substr of your GPU> --n 200000 --frames 30
```

## Full pipeline (stereo 544x480, settled real scene, 1.59M gaussians)

| GPU | full tick | host CPU | notes |
|---|---|---|---|
| NVIDIA RTX 5070 Laptop | 14.7 ms (OneSweep) | ~19% single-core | GPU-bound; host submit-only is 0.67 ms |
| Intel Arc 140T iGPU | 104 ms (hierarchical scan) | ~41% single-core | first proof of cross-vendor, unoptimized |

Host-side per-frame breakdown on the 5070: empty submit 0.039 ms,
frame submit-only 0.67 ms, two `map_read` readbacks dominate the
GPU-serial tail. Rate-limited to 15 Hz the process costs 7-9% of one
core — cheap enough to sit beside a 200 Hz controller.

## Correctness across vendors

Same WGSL, same inputs (1.59M gaussians), NVIDIA vs Intel:

| metric | value | gate |
|---|---|---|
| RGB mean abs diff | 0.0003 / 255 | < 1 |
| RGB max diff | 1 / 255 | — |
| RGB ≤1/255 share | 100% | >99% ≤2 |
| depth median / mean error | 0.0 / 0.01 mm | ≤2 mm |
| intersection count | 3,203,909 vs 3,203,911 | ±2 from exp/ln ulp at a ceil boundary |

The ±2 isect difference is a float-ulp effect at a `ceil()` boundary on
two gaussians, not an algorithmic divergence. Every pixel of both RGBs is
within 1/255.

## Known device quirks (handled, not hidden)

- **OneSweep look-back deadlocks Mesa/Intel** (device lost, even with 2
  blocks) → `pick_variant()` selects the hierarchical-scan sort on any
  non-NVIDIA vendor; all other optimizations stay.
- **Small dispatches abort on Intel ARL ANV** (n ≤ 20k panics; 200k is
  fine) → the bench default uses the proven n=200k scene so one config
  passes on every vendor.
- Resolution is fully runtime-parameterized; the (gaussian, tile) pair
  capacity of 4M covers up to ~800x720 at that scene density (1088x960
  needs 8M).

## Scaling notes

Radix sort dominates GPU time (5G shared-memory reads at 3.2M pairs for
the 6-pass rank count). Known optimization paths if this ever matters
more than portability: 16-bit digits, decoupled look-back, CAP converged
to actual n, staging-ring readback.
