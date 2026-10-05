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
        """Read a 3DGS PLY returning the RAW stored values (log-scale,
        logit-opacity) — apply exp/sigmoid before feeding a Pipeline, whose
        kernels consume activated values. Two layouts are auto-detected from
        the vertex element: the standard INRIA layout, and the super-splat
        compressed layout (quantized `chunk` + `packed_*` uint32 properties;
        chunked min/max bounds, 11/10/11-bit positions and scales, 8-bit
        colors/opacity, largest-component-index rotations). Needs the [io]
        extra (plyfile)."""
        try:
            from plyfile import PlyData
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "GaussianCloud.from_ply needs a PLY reader: "
                "pip install gausscam[io]"
            ) from exc
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        names = {p.name for p in v.properties}
        if "packed_position" in names:
            return cls._from_super_splat(ply, path)
        return cls._from_inria(v, path)

    @classmethod
    def _from_inria(cls, v, path) -> "GaussianCloud":
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

    @classmethod
    def _from_super_splat(cls, ply, path) -> "GaussianCloud":
        """Decode the super-splat compressed layout back to RAW values: the
        container stores ACTIVATED scale/opacity (exp/sigmoid applied at
        encode time) quantized against per-chunk min/max, so scale goes back
        through ln and opacity through logit."""
        SH_C0 = 0.28209479177387814
        vtx = ply["vertex"].data
        chk = ply["chunk"].data
        n = len(vtx)
        if n == 0:
            return cls(np.zeros((0, 3), np.float32),
                       np.zeros((0, 4), np.float32),
                       np.zeros((0, 3), np.float32),
                       np.zeros((0,), np.float32),
                       np.zeros((0, 3), np.float32))
        ci = np.clip(np.arange(n, dtype=np.int64) // 256, 0, len(chk) - 1)

        def chunk(field: str) -> np.ndarray:
            return np.asarray(chk[field])[ci]

        # positions: 11/10/11 bits against per-chunk float bounds (values)
        pp = vtx["packed_position"].astype(np.uint32)
        xyz = np.stack([
            ((pp >> 21) & 0x7FF).astype(np.float32) / 2047.0
            * (chunk("max_x") - chunk("min_x")) + chunk("min_x"),
            ((pp >> 11) & 0x3FF).astype(np.float32) / 1023.0
            * (chunk("max_y") - chunk("min_y")) + chunk("min_y"),
            (pp & 0x7FF).astype(np.float32) / 2047.0
            * (chunk("max_z") - chunk("min_z")) + chunk("min_z"),
        ], axis=1)

        # scales: same 11/10/11 layout against LOG-scale bounds; the decoded
        # values are already the RAW log-scale the dataclass contract wants
        ps = vtx["packed_scale"].astype(np.uint32)
        scale = np.stack([
            ((ps >> 21) & 0x7FF).astype(np.float32) / 2047.0
            * (chunk("max_scale_x") - chunk("min_scale_x")) + chunk("min_scale_x"),
            ((ps >> 11) & 0x3FF).astype(np.float32) / 1023.0
            * (chunk("max_scale_y") - chunk("min_scale_y")) + chunk("min_scale_y"),
            (ps & 0x7FF).astype(np.float32) / 2047.0
            * (chunk("max_scale_z") - chunk("min_scale_z")) + chunk("min_scale_z"),
        ], axis=1)

        # colors: 8-bit against per-chunk bounds in SH-DC-RGB space
        # (f_dc * SH_C0 + 0.5); invert to RAW f_dc. alpha carries the
        # ACTIVATED opacity -> logit back to RAW.
        pc = vtx["packed_color"].astype(np.uint32)
        rgb = np.stack([
            ((pc >> 24) & 0xFF).astype(np.float32) / 255.0
            * (chunk("max_r") - chunk("min_r")) + chunk("min_r"),
            ((pc >> 16) & 0xFF).astype(np.float32) / 255.0
            * (chunk("max_g") - chunk("min_g")) + chunk("min_g"),
            ((pc >> 8) & 0xFF).astype(np.float32) / 255.0
            * (chunk("max_b") - chunk("min_b")) + chunk("min_b"),
        ], axis=1)
        f_dc = (rgb - 0.5) / SH_C0
        p_opa = ((pc & 0xFF).astype(np.float32) / 255.0).clip(1e-6, 1.0 - 1e-6)
        opacity = np.log(p_opa / (1.0 - p_opa))

        # rotations: largest-component index (2 bits) + 3x10-bit magnitudes
        # for the remaining components, in wxyz order; the largest component
        # is reconstructed positive-normalized (the encode side flips sign
        # so this loses only a global sign, which does not affect rotation)
        pr = vtx["packed_rotation"].astype(np.uint32)
        largest = (pr >> 30) & 0x3
        vals = np.stack([(pr >> 20) & 0x3FF, (pr >> 10) & 0x3FF,
                         pr & 0x3FF], axis=1).astype(np.float32)
        vals = (vals / 1023.0 - 0.5) / (np.sqrt(2.0) * 0.5)
        q = np.zeros((n, 4), np.float32)
        m = [largest == k for k in range(4)]
        comp = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
        for k in range(4):
            for slot, axis in enumerate(comp[k]):
                q[m[k], axis] = vals[m[k], slot]
        rest_sq = np.sum(q * q, axis=1)
        big = np.sqrt(np.clip(1.0 - rest_sq, 0.0, 1.0)).astype(np.float32)
        for k in range(4):
            q[m[k], k] = big[m[k]]

        # optional high-order SH element: uint8 planar (R..., G..., B...),
        # de-quantized to (u8 / 256 - 0.5) * 8 and re-interleaved to the
        # dataclass (RGB, RGB...) layout; DC comes from packed_color above
        f_rest = np.zeros((n, 0), np.float32)
        if "sh" in [e.name for e in ply.elements]:
            sh_el = ply["sh"]
            rest_names = [p.name for p in sh_el.properties
                          if p.name.startswith("f_rest_")]
            if rest_names:
                raw = np.stack([np.asarray(sh_el[nm]) for nm in rest_names],
                               axis=1).astype(np.float32)
                f_rest = ((raw / 256.0 - 0.5) * 8.0).reshape(n, 3, -1)
                f_rest = f_rest.transpose(0, 2, 1).reshape(n, -1)
        sh = np.ascontiguousarray(np.concatenate([f_dc, f_rest], axis=1))

        return cls(
            np.ascontiguousarray(xyz, np.float32),
            np.ascontiguousarray(q, np.float32),
            np.ascontiguousarray(scale, np.float32),
            np.ascontiguousarray(opacity, np.float32),
            sh,
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
