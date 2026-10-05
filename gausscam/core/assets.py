"""Splat assets: a static scene cloud + per-link robot clouds (numpy only).

Robot gaussians are stored per link (one rigid blob per kinematic link) so the
backend can move them with the simulator's link poses instead of re-rendering
a mesh. Layout convention shared with gsrender_lite: the merged buffer is
scene-first, robot-block-last, and each robot gaussian carries a link slot id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GaussianCloud:
    """One gaussian blob. All arrays float32 contiguous, row-major."""

    xyz: np.ndarray  # [N, 3]
    rot: np.ndarray  # [N, 4] quaternion wxyz
    scale: np.ndarray  # [N, 3]
    opacity: np.ndarray  # [N]
    sh: np.ndarray  # [N, K*3] flat SH coefficients, K = (degree+1)^2

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def sh_degree(self) -> int:
        k = self.sh.shape[1] // 3
        return int(round(k**0.5)) - 1

    @classmethod
    def from_ply(cls, path: str | Path) -> "GaussianCloud":
        """Read a 3DGS PLY (INRIA layout) returning the RAW stored values
        (log-scale, logit-opacity) — apply exp/sigmoid before feeding a
        Pipeline, whose kernels consume activated values. Needs the [io]
        extra (plyfile)."""
        try:
            from plyfile import PlyData
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "GaussianCloud.from_ply needs a PLY reader: "
                "pip install gausscam[io]"
            ) from exc
        v = PlyData.read(str(path))["vertex"]
        names = {p.name for p in v.properties}

        def prop(*keys: str) -> np.ndarray:
            missing = [k for k in keys if k not in names]
            if missing:
                raise ValueError(f"{path}: PLY missing properties {missing}")
            return np.stack([np.asarray(v[k]) for k in keys], axis=-1)

        rest = sorted(
            (n for n in names if n.startswith("f_rest_")),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )
        sh = np.concatenate(
            [prop("f_dc_0", "f_dc_1", "f_dc_2"),
             prop(*rest) if rest else np.zeros((len(v.data), 0), np.float32)],
            axis=-1,
        )
        return cls(
            np.ascontiguousarray(prop("x", "y", "z"), np.float32),
            np.ascontiguousarray(prop("rot_0", "rot_1", "rot_2", "rot_3"),
                                 np.float32),
            np.ascontiguousarray(prop("scale_0", "scale_1", "scale_2"),
                                 np.float32),
            np.ascontiguousarray(np.asarray(v["opacity"]), np.float32),
            np.ascontiguousarray(sh, np.float32),
        )


@dataclass
class SceneSplat:
    """Static environment splat (one ply, e.g. a captured LIO-mapped scene)."""

    cloud: GaussianCloud

    @classmethod
    def load(cls, ply_path: str | Path) -> "SceneSplat":
        return cls(GaussianCloud.from_ply(ply_path))


@dataclass
class RobotSplat:
    """Per-link robot splats.

    `link_names` order IS the backend link-slot order (slot 0 = link_names[0]).
    The names should match the simulator's link/body names one-to-one so the
    adapter can feed poses without a mapping table (add a static offset table
    here later if a rig mismatch ever shows up as visual clipping).
    """

    link_names: list[str]
    clouds: dict[str, GaussianCloud]

    @classmethod
    def load_dir(
        cls, robot_gs_dir: str | Path, link_names: list[str] | None = None
    ) -> "RobotSplat":
        d = Path(robot_gs_dir)
        names = link_names if link_names is not None else sorted(
            p.stem for p in d.glob("*.ply")
        )
        missing = [n for n in names if not (d / f"{n}.ply").is_file()]
        if missing:
            raise FileNotFoundError(f"{d}: missing link plys {missing}")
        return cls(list(names), {n: GaussianCloud.from_ply(d / f"{n}.ply") for n in names})

    @property
    def n_gaussians(self) -> int:
        return sum(c.n for c in self.clouds.values())


def merge(
    scene: SceneSplat, robot: RobotSplat | None
) -> tuple[GaussianCloud, np.ndarray | None, int]:
    """Merge scene-first / robot-last (gsrender_lite robot block = the tail).

    Returns (cloud, slots, n_links): slots[i] is the link slot of robot
    gaussian i (None when robot is None). All clouds must share one SH degree;
    the caller falls back to degree 0 clouds if assets disagree.
    """
    if robot is None:
        return scene.cloud, None, 0
    parts = [scene.cloud] + [robot.clouds[n] for n in robot.link_names]
    deg = min(p.sh_degree for p in parts)
    if len({p.sh_degree for p in parts}) > 1:
        # e.g. dc-only robot link plys alongside a degree-3 scene: fall back
        # to the common degree (the first 3 SH coeffs = f_dc)
        parts = [
            GaussianCloud(p.xyz, p.rot, p.scale, p.opacity,
                          np.ascontiguousarray(p.sh[:, :3 * (deg + 1)]))
            for p in parts
        ]
    merged = GaussianCloud(
        xyz=np.concatenate([p.xyz for p in parts]),
        rot=np.concatenate([p.rot for p in parts]),
        scale=np.concatenate([p.scale for p in parts]),
        opacity=np.concatenate([p.opacity for p in parts]),
        sh=np.concatenate([p.sh for p in parts]),
    )
    slot_of = {n: i for i, n in enumerate(robot.link_names)}
    slots = np.concatenate(
        [np.full(robot.clouds[n].n, slot_of[n], np.int32) for n in robot.link_names]
    )
    return merged, slots, len(robot.link_names)
