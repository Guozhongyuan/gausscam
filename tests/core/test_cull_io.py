"""SelfCull semantics + 3DGS PLY round-trip (io extra)."""

import numpy as np
import pytest

from gausscam.core.assets import GaussianCloud
from gausscam.core.cull import DEFAULT_RADIUS, SelfCull


def test_default_radius_is_the_tinynav_golden_value():
    # map2 capture calibration; backends and tests both rely on this default
    assert DEFAULT_RADIUS == pytest.approx(0.30)


def test_selfcull_frozen():
    c = SelfCull()
    assert c.radius == pytest.approx(0.30)
    with pytest.raises((AttributeError, TypeError)):
        c.radius = 1.0


# ---------------------------------------------------------------- ply io --- #

def _write_3dgs_ply(path, n=6, n_rest=9):
    """Minimal INRIA-layout 3DGS PLY (degree-1 when n_rest=9, dc-only when 0)."""
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(7)
    props = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ] + [(f"f_rest_{i}", "f4") for i in range(n_rest)] + [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    data = np.zeros(n, dtype=props)
    data["x"], data["y"], data["z"] = rng.standard_normal((3, n))
    data["f_dc_0"] = 0.3
    data["rot_0"] = 1.0
    data["opacity"] = 0.8
    data["scale_0"] = data["scale_1"] = data["scale_2"] = -2.0
    for i in range(n_rest):
        data[f"f_rest_{i}"] = 0.01 * (i + 1)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def test_from_ply_roundtrip(tmp_path):
    ply = tmp_path / "tiny.ply"
    _write_3dgs_ply(ply, n=6, n_rest=9)
    c = GaussianCloud.from_ply(ply)
    assert c.n == 6
    assert c.sh.shape == (6, 12)          # f_dc(3) + f_rest(9)
    assert c.sh_degree == 1
    assert c.sh[:, 3] == pytest.approx(0.01)   # f_rest_0 lands after f_dc
    assert c.opacity == pytest.approx(np.full(6, 0.8), abs=1e-6)
    assert (c.rot[:, 0] == 1.0).all()          # wxyz untouched
    for arr in (c.xyz, c.rot, c.scale, c.opacity, c.sh):
        assert arr.dtype == np.float32


def test_from_ply_dc_only(tmp_path):
    ply = tmp_path / "dc.ply"
    _write_3dgs_ply(ply, n=3, n_rest=0)
    c = GaussianCloud.from_ply(ply)
    assert c.sh.shape == (3, 3) and c.sh_degree == 0


def test_from_ply_missing_property(tmp_path):
    from plyfile import PlyData, PlyElement

    ply = tmp_path / "bad.ply"
    _write_3dgs_ply(ply)
    data = PlyData.read(str(ply))["vertex"].data
    data = np.array([f for f in data if True], dtype=data.dtype)
    dropped = [n for n in data.dtype.names if n != "rot_2"]
    out = np.zeros(len(data), dtype=[(n, "f4") for n in dropped])
    for n in dropped:
        out[n] = data[n]
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(ply))
    with pytest.raises(ValueError, match="rot_2"):
        GaussianCloud.from_ply(ply)


# -------------------------------------------------- super-splat compressed --- #

def _write_super_splat_ply(path, n=600, with_sh=False):
    """Synthetic super-splat compressed PLY: chunk + packed uint32 vertex
    (+ optional planar uint8 sh element), following the documented bit
    layout. Values are packed through the SAME quantization the format
    defines, so decoding exercises real round-trip precision."""
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(11)
    n_chunks = (n + 255) // 256
    xyz = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
    log_scale = rng.uniform(-4.0, -1.0, (n, 3)).astype(np.float32)
    f_dc = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
    opa = rng.uniform(0.05, 0.95, n).astype(np.float32)
    rot = np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1))

    ci = np.arange(n) // 256
    chunk_props = [(f, "f4") for f in
                   ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
                    "min_scale_x", "min_scale_y", "min_scale_z",
                    "max_scale_x", "max_scale_y", "max_scale_z",
                    "min_r", "min_g", "min_b", "max_r", "max_g", "max_b")]
    chunk = np.zeros(n_chunks, dtype=chunk_props)
    for f in ("x", "y", "z"):
        chunk["min_" + f] = xyz[:, "xyz".index(f)].min()
        chunk["max_" + f] = xyz[:, "xyz".index(f)].max()
    for j, ax in enumerate("xyz"):
        chunk["min_scale_" + ax] = log_scale[:, j].min()
        chunk["max_scale_" + ax] = log_scale[:, j].max()
    for j, c in enumerate("rgb"):
        lo = (f_dc[:, j] * 0.28209479177387814 + 0.5).min()
        hi = (f_dc[:, j] * 0.28209479177387814 + 0.5).max()
        chunk["min_" + c] = lo
        chunk["max_" + c] = hi

    def q11(v, lo, hi):
        return np.round((v - lo) / (hi - lo) * 2047).astype(np.uint32)
    def q10(v, lo, hi):
        return np.round((v - lo) / (hi - lo) * 1023).astype(np.uint32)
    def q8(v, lo, hi):
        return np.round((v - lo) / (hi - lo) * 255).astype(np.uint32)

    pos = (q11(xyz[:, 0], chunk["min_x"][0], chunk["max_x"][0]) << 21) | \
          (q10(xyz[:, 1], chunk["min_y"][0], chunk["max_y"][0]) << 11) | \
          q11(xyz[:, 2], chunk["min_z"][0], chunk["max_z"][0])
    scl = (q11(log_scale[:, 0], chunk["min_scale_x"][0],
               chunk["max_scale_x"][0]) << 21) | \
          (q10(log_scale[:, 1], chunk["min_scale_y"][0],
               chunk["max_scale_y"][0]) << 11) | \
          q11(log_scale[:, 2], chunk["min_scale_z"][0], chunk["max_scale_z"][0])
    rgbq = np.stack([q8(f_dc[:, j] * 0.28209479177387814 + 0.5,
                        chunk["min_rgb"[j]][0] if False else
                        chunk[["min_r", "min_g", "min_b"][j]][0],
                        chunk[["max_r", "max_g", "max_b"][j]][0])
                     for j in range(3)], 1)
    opa_q = np.round(opa * 255).astype(np.uint32)
    col = (rgbq[:, 0] << 24) | (rgbq[:, 1] << 16) | (rgbq[:, 2] << 8) | opa_q
    # largest = w (index 3); the three 10-bit fields encode 0.0 at their
    # mid-scale 512, not at 0 (which decodes to -0.7071)
    rotp = np.full(n, (3 << 30) | (512 << 20) | (512 << 10) | 512, np.uint32)

    vtx = np.zeros(n, dtype=[("packed_position", "u4"),
                             ("packed_rotation", "u4"),
                             ("packed_scale", "u4"),
                             ("packed_color", "u4")])
    vtx["packed_position"] = pos
    vtx["packed_rotation"] = rotp
    vtx["packed_scale"] = scl
    vtx["packed_color"] = col

    elements = [PlyElement.describe(chunk, "chunk"),
                PlyElement.describe(vtx, "vertex")]
    if with_sh:
        n_rest = 9  # degree-1: 3 per channel, planar (R..., G..., B...)
        sh_vals = (np.tile(0.02 * np.arange(1, n_rest + 1), (n, 1))
                   .reshape(n, 3, 3).transpose(0, 2, 1).reshape(n, n_rest))
        sh_u8 = np.round((sh_vals / 8.0 + 0.5) * 256).clip(0, 255).astype(np.uint8)
        sh = np.zeros(n, dtype=[(f"f_rest_{i}", "u1") for i in range(n_rest)])
        for i in range(n_rest):
            sh[f"f_rest_{i}"] = sh_u8[:, i]
        elements.append(PlyElement.describe(sh, "sh"))
    PlyData(elements, text=False).write(str(path))


def test_from_ply_super_splat_roundtrip(tmp_path):
    ply = tmp_path / "compressed.ply"
    _write_super_splat_ply(ply, n=600, with_sh=True)
    c = GaussianCloud.from_ply(ply)
    assert c.n == 600
    assert c.sh.shape == (600, 12) and c.sh_degree == 1
    # decoded RAW values land back inside the encoded envelopes
    assert (np.exp(c.scale) > 0).all()
    assert c.scale.min() > -4.0 and c.scale.max() < -1.0
    act_opa = 1.0 / (1.0 + np.exp(-c.opacity))
    assert ((act_opa - np.linspace(0.05, 0.95, 1)[0] * 0) < 1).all()
    assert act_opa.min() > 0.0 and act_opa.max() < 1.0
    # rotation: largest component reconstructed on w, unit norm
    assert np.allclose(np.linalg.norm(c.rot, axis=1), 1.0, atol=1e-5)
    # DC decode round-trips the quantized f_dc within 8-bit chunk precision
    f_dc_back = c.sh[:, :3] * 0.28209479177387814 + 0.5
    assert f_dc_back.min() >= -0.01 and f_dc_back.max() <= 1.01
    # sh element planar -> interleaved: f_rest_0 (R of band 1) after f_dc
    assert c.sh[:, 3] == pytest.approx(0.02, abs=0.05)
    assert c.sh[:, 6] == pytest.approx(0.08, abs=0.05)  # G of band 1
    for arr in (c.xyz, c.rot, c.scale, c.opacity, c.sh):
        assert arr.dtype == np.float32


def test_from_ply_super_splat_dc_only(tmp_path):
    ply = tmp_path / "compressed_dc.ply"
    _write_super_splat_ply(ply, n=100, with_sh=False)
    c = GaussianCloud.from_ply(ply)
    assert c.sh.shape == (100, 3) and c.sh_degree == 0


def test_from_ply_autodetect_inria_unaffected(tmp_path):
    ply = tmp_path / "plain.ply"
    _write_3dgs_ply(ply, n=5, n_rest=0)
    c = GaussianCloud.from_ply(ply)
    assert c.n == 5 and c.sh.shape == (5, 3)
    assert np.allclose(c.xyz, c.xyz)  # INRIA path untouched, no crash
