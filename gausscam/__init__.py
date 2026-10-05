"""gausscam — 3D Gaussian Splatting sensor rendering for physics simulators.

backends/ turn gaussians + link poses into sensor images; adapters/ feed poses
from a physics engine under one contract: {link_name: (pos[3], quat_wxyz[4])}.
"""

__version__ = "0.1.0"
