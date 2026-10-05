"""RenderBackend protocol — the only surface backends and callers share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One rendered sensor frame.

    depth is expected-z along the camera axis in millimetres (uint16, 0 = no
    hit within the far plane) — same semantics as the tinynav sensor ring.
    """

    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] uint16, mm

    @property
    def depth_m(self) -> np.ndarray:
        return self.depth.astype(np.float32) / 1000.0


@runtime_checkable
class RenderBackend(Protocol):
    def upload(self, scene, robot=None) -> None:
        """Upload a merged scene+robot splat (see core.assets.merge)."""
        ...

    def update_links(self, poses: Mapping[str, tuple]) -> None:
        """Feed link poses: {link_name: (pos[3], quat_wxyz[4])}, f32."""
        ...

    def render(
        self,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy_deg: float,
        width: int,
        height: int,
        *,
        cull_origin: np.ndarray | None = None,
        cull_radius: float = 0.30,
    ) -> Sequence[Frame]:
        """Render all cameras in one batch; MuJoCo camera convention
        (cam_xmat = camera-to-world, x right / y up / -z forward)."""
        ...
