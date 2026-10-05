# AGENTS.md — gausscam

Rules for AI agents (and humans) changing this repository.

## Language

- All code comments, docstrings, commit messages, README and docs are
  written in **English**. Do not introduce Chinese (or other non-English)
  text anywhere in the repo.

## Tests

- Every behavior change or bug fix ships with a test. `tests/` is the
  contract: `pytest -m "not gpu"` must pass without any GPU; GPU-marked
  tests run when a Vulkan adapter is visible.
- Prefer `gausscam.synthetic.make_scene` over captured assets in tests —
  the repo must stay installable and testable with zero downloads.
- Tests that touch wgpu run in a subprocess (`test_bench_smoke.py`
  pattern): a driver panic must abort the test, not the pytest process.

## Performance

- This library exists to be cheap on the host. Regressions matter as much
  as bugs: after changing a hot path (`Pipeline.render_frame`,
  `set_links`/`set_cam`, `cull`, per-tick adapter calls), measure before
  and after with `python -m gausscam.bench` and report the delta in the
  description.
- No per-tick allocations or name lookups in code called at controller
  rate (200 Hz); cache at construction (see `MuJoCoAdapter._body_ids`).
- The GPU axis has two sort variants; keep both working and keep
  `pick_variant()` the only place that chooses.

## Repo boundaries

- `examples/` is NOT part of the package and is git-ignored; never import
  from it in `gausscam/` or `tests/`. The package must not reference any
  asset file (npz/ply) — synthetic data or user-supplied inputs only.
- Public API is `Pipeline`, `pick_variant`, `build_wgsl`, `make_K` plus
  `gausscam.core.*` and `gausscam.synthetic`. Mark internal helpers with a
  leading underscore; do not widen `__all__` casually.
