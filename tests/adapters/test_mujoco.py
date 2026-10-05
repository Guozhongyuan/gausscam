"""MuJoCo adapter: pose contract against mjData ground truth."""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from gausscam.adapters.mujoco import MuJoCoAdapter  # noqa: E402

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body name="base" pos="0.1 0.2 0.3" euler="0 0 1.5708">
      <freejoint/>
      <geom type="box" size=".05 .05 .05"/>
      <camera name="cam0" pos="0.19 0.02 0.09"/>
      <body name="wing" pos="0.1 0 0">
        <geom type="sphere" size=".02"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture()
def adapter():
    model = mujoco_model()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return MuJoCoAdapter(model, data), model, data


def mujoco_model():
    return mj.MjModel.from_xml_string(XML)


def test_body_poses_match_mjdata(adapter):
    ad, model, data = adapter
    poses = ad.body_poses(["base", "wing"])
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
    pos, quat = poses["base"]
    assert pos.dtype == np.float32 and quat.dtype == np.float32
    assert np.allclose(pos, data.xpos[bid], atol=1e-6)
    assert np.allclose(quat, data.xquat[bid], atol=1e-6)   # wxyz native
    assert "wing" in poses


def test_body_poses_unknown_name(adapter):
    ad, _, _ = adapter
    with pytest.raises(KeyError, match="nope"):
        ad.body_poses(["nope"])


def test_camera_poses_match_mjdata(adapter):
    ad, model, data = adapter
    pos, xmat = ad.camera_poses(["cam0"])
    cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "cam0")
    assert pos.shape == (1, 3) and xmat.shape == (1, 3, 3)
    assert np.allclose(pos[0], data.cam_xpos[cid], atol=1e-6)
    assert np.allclose(xmat[0], data.cam_xmat[cid].reshape(3, 3), atol=1e-6)


def test_poses_track_physics(adapter):
    ad, model, data = adapter
    before = ad.body_poses(["base"])["base"][0].copy()
    data.qvel[0] = 0.5                       # push along x
    for _ in range(20):
        mj.mj_step(model, data)
    after = ad.body_poses(["base"])["base"][0]
    assert not np.allclose(before, after)
    assert after[0] > before[0]
