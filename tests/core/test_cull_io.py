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
