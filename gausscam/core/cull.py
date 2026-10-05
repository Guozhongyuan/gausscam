"""Sensor semantics that live in core but are implemented per backend.

SelfCull: hide the robot's own splats within `radius` of a sensor origin.
Per-link splats are fatter than the real limbs, so a camera mounted on the
body would sit inside torso/leg gaussians and see a permanent out-of-focus
blob over half the frame. Backends zero robot-gaussian opacity in the sphere
(device-side); nothing accumulates across frames.
"""

from __future__ import annotations

from dataclasses import dataclass

# Golden value carried over from the tinynav sensor stack (map2 capture).
DEFAULT_RADIUS = 0.30


@dataclass(frozen=True)
class SelfCull:
    radius: float = DEFAULT_RADIUS
