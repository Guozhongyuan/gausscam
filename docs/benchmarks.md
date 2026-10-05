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

## Scene size (adaptive capacity, RTX 5070 Laptop)

The (gaussian, tile) pair capacity scales with the cloud —
`auto_capacity(N) = clamp(4·N, 4M, 64M)`, override with
`Pipeline(cap=...)` — and a pair-count tripwire raises on overflow
instead of dropping splats silently. Single 640x544 pinhole, 10 Hz loop:

| scene | gaussians | cap | sort scratch | GPU latency | host CPU @ 10 Hz |
|---|---|---|---|---|---|
| map3 stairwell (DC-only PLY) | 983k | 4M | ~114 MB | 9.4 ms | 5% of one core |
| church nave (full cloud) | 7.67M | 30.7M | ~878 MB | 48.3 ms | 5% of one core |

Host cost is scene-size independent (submit + two readbacks); GPU
latency grows sublinearly with N (7.8x the gaussians, 5.1x the frame
time). Pipeline init stays ~1 s at both sizes.

### Before/after the adaptive capacity (0.1.1 vs 0.1.2)

Same machine, back-to-back runs, single 640x544 pinhole, medians of
repeated 40-frame batches (the HIL simulator was rendering on the same
GPU throughout — absolute numbers drift a few percent, the deltas held):

| case | 0.1.1 (fixed 4M cap) | 0.1.2 (adaptive cap) |
|---|---|---|
| map3 983k — fits the 4M cap | 8.4 ms/frame | 9.4 ms/frame |
| church 2M subsample — fits the 4M cap | 14.5 ms/frame | 14.6 ms/frame |
| church 7.67M full cloud | 92.8 ms/frame, **45.7% of pixels silently dropped** | 48.3 ms/frame, complete; overflow raises |

Two findings. First, the price of the safety machinery on scenes that
never overflow is ~0.9 ms/frame at a 4M cap (one extra 4-byte pair-count
readback per frame plus grid-stride scatter) — under 1% of a 10 Hz frame
budget; 2M-gaussian clouds are unaffected. Second, an overflowing scene
on the fixed cap was not just incomplete but SLOWER (92.8 vs 48.3 ms):
the rasterizer keeps churning through the dropped pairs' tile ranges.
Sizing the capacity to the cloud fixed both.

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
  capacity auto-scales with the cloud (`auto_capacity(N)`, clamp
  [4M, 64M], `Pipeline(cap=...)` overrides) and overflowing it raises.

## Scaling notes

Radix sort dominates GPU time (5G shared-memory reads at 3.2M pairs for
the 6-pass rank count). Known optimization paths if this ever matters
more than portability: 16-bit digits, decoupled look-back, CAP converged
to actual n, staging-ring readback.
