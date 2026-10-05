"""SensorRig / SensorCam contract tests (pure numpy)."""

import numpy as np
import pytest

from gausscam.core.rig import SensorCam, SensorRig, quat_from_xyaxes, quat_to_xmat


def test_K_from_fovy():
    cam = SensorCam("c", width=640, height=480, fovy_deg=90.0,
                    pos=np.zeros(3), quat_wxyz=np.array([1.0, 0, 0, 0]))
    K = cam.K()
    # fovy 90 -> fy = H/2 / tan(45) = H/2 ; principal point at image center
    assert K[0, 0] == pytest.approx(240.0)
    assert K[1, 1] == pytest.approx(240.0)
    assert K[0, 2] == pytest.approx(320.0)
    assert K[1, 2] == pytest.approx(240.0)


def test_K_roundtrip_fovy():
    fovy = 82.85
    cam = SensorCam("c", 544, 480, fovy, np.zeros(3),
                    np.array([1.0, 0, 0, 0]))
    fy = cam.K()[1, 1]
    back = np.degrees(2 * np.arctan((480 / 2) / fy))
    assert back == pytest.approx(fovy, abs=1e-6)


def test_d435i_rig_layout():
    rig = SensorRig.d435i()
    assert rig.names == ["infra1", "infra2", "color"]
    i1, i2, color = rig.cams
    assert abs(i1.pos[1]) == pytest.approx(abs(i2.pos[1]))  # symmetric
    assert i1.pos[1] > 0 > i2.pos[1]
    assert np.allclose(color.pos[:1], i1.pos[:1])
    # fy=272 at H=480 -> fovy ~82.85 deg
    assert i1.fovy_deg == pytest.approx(82.85, abs=0.1)


def test_cam_poses_identity_parent():
    rig = SensorRig.d435i(baseline=0.06)
    pos, xmat = rig.cam_poses(np.zeros(3), np.eye(3))
    assert pos.shape == (3, 3) and xmat.shape == (3, 3, 3)
    assert pos.dtype == np.float32 and xmat.dtype == np.float32
    assert np.allclose(pos[0], rig.cams[0].pos, atol=1e-6)


def test_cam_poses_compose_with_parent_rotation():
    """Composition = parent R applied to the cam offset; verify with a +90
    deg z-yaw matrix (independently covered by quat_to_xmat tests)."""
    rig = SensorRig.d435i()
    yaw90 = quat_to_xmat(np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
    p = np.array([1.0, 2.0, 3.0])
    pos, _ = rig.cam_poses(p, yaw90)
    for cam, got in zip(rig.cams, pos):
        want = yaw90 @ np.asarray(cam.pos, np.float64) + p
        assert np.allclose(got, want, atol=1e-5)


def test_quat_from_xyaxes_columns():
    """quat_from_xyaxes(x, y) -> rotation whose body x/y axes map to x/y."""
    R = quat_to_xmat(quat_from_xyaxes([0, -1, 0], [0, 0, 1]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-6)   # x axis
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-6)    # y axis


def test_identity_quat():
    assert np.allclose(quat_to_xmat(np.array([1.0, 0, 0, 0])), np.eye(3))
