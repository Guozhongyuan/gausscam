"""MuJoCo adapter — pull poses straight out of mjData.

MuJoCo stores body quaternions as wxyz natively, so the gausscam pose
contract needs zero conversion. Sensor cameras can be declared on the robot
body in the MJCF (recommended: the scene stays the single source of truth)
and read back via camera_poses(); or computed from a SensorRig + the parent
link pose via rig.cam_poses().
"""

from __future__ import annotations

import numpy as np

import mujoco as mj


class MuJoCoAdapter:
    def __init__(self, model: "mj.MjModel", data: "mj.MjData"):
        self.model = model
        self.data = data
        # name->id caches: mj_name2id is a string scan; poses are read at
        # control rate (200 Hz) and the cache turns that into array indexing.
        self._body_ids: dict[str, int] = {}
        self._cam_ids: dict[str, int] = {}

    def body_poses(self, names) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """{name: (xpos[3], xquat[4] wxyz)} for the given body names."""
        out = {}
        for n in names:
            bid = self._body_ids.get(n)
            if bid is None:
                bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, n)
                if bid < 0:
                    raise KeyError(f"no body named {n!r} in model")
                self._body_ids[n] = bid
            out[n] = (
                np.asarray(self.data.xpos[bid], np.float32),
                np.asarray(self.data.xquat[bid], np.float32),
            )
        return out

    def camera_poses(self, names) -> tuple[np.ndarray, np.ndarray]:
        """(pos [C,3], xmat [C,3,3]) for the given camera names."""
        pos, xmat = [], []
        for n in names:
            cid = self._cam_ids.get(n)
            if cid is None:
                cid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, n)
                if cid < 0:
                    raise KeyError(f"no camera named {n!r} in model")
                self._cam_ids[n] = cid
            pos.append(self.data.cam_xpos[cid])
            xmat.append(self.data.cam_xmat[cid])
        return (
            np.asarray(pos, np.float32),
            np.asarray(xmat, np.float32).reshape(-1, 3, 3),
        )
