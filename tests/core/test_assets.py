"""Synthetic 3DGS assets for tests — deterministic, no external files."""

import numpy as np

from gausscam.core.assets import GaussianCloud, RobotSplat, SceneSplat, merge


def make_cloud(n=8, seed=0, sh_width=3, xyz=None, scale=None, opacity=None):
    rng = np.random.default_rng(seed)
    return GaussianCloud(
        xyz=xyz if xyz is not None else rng.standard_normal((n, 3)).astype(np.float32),
        rot=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1)),
        scale=scale if scale is not None
        else rng.uniform(0.01, 0.05, (n, 3)).astype(np.float32),
        opacity=opacity if opacity is not None
        else rng.uniform(0.5, 1.0, (n,)).astype(np.float32),
        sh=rng.standard_normal((n, sh_width)).astype(np.float32),
    )


def make_scene_and_robot(n_scene=8, per_link=4, links=("base", "FL_hip")):
    scene = SceneSplat(make_cloud(n_scene, seed=1))
    robot = RobotSplat(
        list(links),
        {name: make_cloud(per_link, seed=2 + i) for i, name in enumerate(links)},
    )
    return scene, robot


def test_cloud_shape_contract():
    c = make_cloud(n=5, sh_width=48)
    assert c.n == 5
    assert c.xyz.shape == (5, 3) and c.rot.shape == (5, 4)
    assert c.scale.shape == (5, 3) and c.opacity.shape == (5,)
    assert c.sh.shape == (5, 48)
    assert c.sh_degree == 3          # 48/3 = 16 = (3+1)^2


def test_cloud_sh_degree_dc_only():
    assert make_cloud(sh_width=3).sh_degree == 0


def test_all_float32():
    c = make_cloud()
    for arr in (c.xyz, c.rot, c.scale, c.opacity, c.sh):
        assert arr.dtype == np.float32


def test_merge_scene_only_passthrough():
    scene, _ = make_scene_and_robot()
    merged, slots, n_links = merge(scene, None)
    assert merged is scene.cloud
    assert slots is None and n_links == 0


def test_merge_scene_first_robot_last():
    scene, robot = make_scene_and_robot(n_scene=8, per_link=4)
    merged, slots, n_links = merge(scene, robot)
    assert n_links == 2
    assert merged.n == 8 + 8
    # scene block occupies [0, 8); robot tails follow in link_names order
    assert (slots == np.array([0] * 4 + [1] * 4, np.int32)).all()
    assert np.allclose(merged.xyz[:8], scene.cloud.xyz)
    assert np.allclose(merged.xyz[8:12], robot.clouds["base"].xyz)
    assert np.allclose(merged.xyz[12:], robot.clouds["FL_hip"].xyz)


def test_merge_downgrades_sh_degree():
    scene = SceneSplat(make_cloud(sh_width=48, seed=3))     # degree 3
    robot = RobotSplat(["base"], {"base": make_cloud(sh_width=3, seed=4)})
    merged, slots, _ = merge(scene, robot)
    assert merged.sh.shape[1] == 3 and merged.sh_degree == 0


def test_merge_dtypes_preserved():
    scene, robot = make_scene_and_robot()
    merged, slots, _ = merge(scene, robot)
    assert merged.xyz.dtype == np.float32
    assert slots.dtype == np.int32
