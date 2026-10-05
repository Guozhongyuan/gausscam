"""SensorRig: sensor cameras attached to a robot link (extrinsics in the
link frame) + the math to resolve world poses for any parent-link pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_from_xyaxes(x, y) -> np.ndarray:
    """Quaternion (wxyz) whose camera frame has the given x (right) and y
    (up) axes — MuJoCo's `xyaxes` camera attribute convention."""
    x = np.asarray(x, np.float64)
    x = x / np.linalg.norm(x)
    y = np.asarray(y, np.float64)
    y = y - (y @ x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    m = np.column_stack([x, y, z])  # columns = camera axes in parent frame
    w = np.sqrt(1.0 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    q = np.array(
        [
            w,
            (m[2, 1] - m[1, 2]) / (4 * w),
            (m[0, 2] - m[2, 0]) / (4 * w),
            (m[1, 0] - m[0, 1]) / (4 * w),
        ]
    )
    return q / np.linalg.norm(q)


@dataclass(frozen=True)
class SensorCam:
    """One sensor camera; extrinsics are in the parent link frame."""

    name: str
    width: int
    height: int
    fovy_deg: float
    pos: np.ndarray  # [3]
    quat_wxyz: np.ndarray  # [4]

    def K(self) -> np.ndarray:
        """Pinhole intrinsics implied by fovy (vertical) at this H."""
        fy = 0.5 * self.height / np.tan(np.radians(0.5 * self.fovy_deg))
        return np.array(
            [[fy, 0, self.width / 2.0], [0, fy, self.height / 2.0], [0, 0, 1.0]],
            np.float64,
        )


@dataclass(frozen=True)
class SensorRig:
    cams: tuple[SensorCam, ...]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cams]

    def cam_poses(
        self, parent_pos: np.ndarray, parent_xmat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """World poses for all cams given the parent link's world pose.

        Returns (pos [C,3] f32, xmat [C,3,3] f32) — directly usable by
        backend.render().
        """
        R = np.asarray(parent_xmat, np.float64).reshape(3, 3)
        p = np.asarray(parent_pos, np.float64)
        pos = np.stack([R @ np.asarray(c.pos, np.float64) + p for c in self.cams])
        xmat = np.stack([R @ quat_to_xmat(c.quat_wxyz) for c in self.cams])
        return pos.astype(np.float32), xmat.astype(np.float32)

    @classmethod
    def d435i(
        cls,
        width: int = 544,
        height: int = 480,
        fy: float = 272.0,
        baseline: float = 0.051,
        with_color: bool = True,
    ) -> "SensorRig":
        """RealSense D435i-style front rack, transcribed from the tinynav
        sensor rig (gs_playground go2_sensor_rig_stairs.xml): all three cams
        look along the parent link's +x, image up = link +z; infra pair
        offset ±baseline/2 laterally. fy=272 @ 480x544 -> fovy 82.85 deg."""
        fovy = float(np.degrees(2.0 * np.arctan((height / 2.0) / fy)))
        q = quat_from_xyaxes([0, -1, 0], [0, 0, 1])
        cams = [
            SensorCam("infra1", width, height, fovy,
                      np.array([0.19, baseline / 2.0, 0.09]), q),
            SensorCam("infra2", width, height, fovy,
                      np.array([0.19, -baseline / 2.0, 0.09]), q),
        ]
        if with_color:
            cams.append(SensorCam("color", width, height, fovy,
                                  np.array([0.19, 0.0, 0.09]), q))
        return cls(tuple(cams))


def quat_to_xmat(q) -> np.ndarray:
    """Quaternion (wxyz) -> 3x3 rotation matrix (camera-to-parent)."""
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
