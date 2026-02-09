from contextlib import contextmanager
from functools import partial, wraps
import pdb
from threading import Event, Thread
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import numpy as np
import imageio

from train import create_model, create_transforms
from datastructs import StereoSample
from modelutils import Transform, box2robot
from rollout_datastructs import Step, Image, SceneObj
from datagen2_isaacsim.isaac_utils import create_empty, set_transform
import utils


def pfind(prim_str, stage):
    from pxr import Usd

    root_prim = stage.GetPrimAtPath("/")
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == prim_str:
            return SceneObj(path=str(prim.GetPath()), prim=prim)

    raise RuntimeError()

def prim_local2world(prim_obj: SceneObj):
    from pxr import UsdGeom, Usd
    prim = prim_obj.prim
    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(matrix).T

def prim_local2parent(prim_obj: SceneObj):
    from pxr import UsdGeom, Usd
    prim = prim_obj.prim
    xform = UsdGeom.Xform(prim)
    matrix = xform.GetLocalTransformation()
    return np.array(matrix).T

def lookat_mat_c2w(origin: np.ndarray, lookat_point: np.ndarray):
    def _norm(vec):
        return vec / np.linalg.norm(vec)

    z_ax = _norm(lookat_point - origin)
    x_ax = _norm(np.cross(z_ax, np.asarray([0, 0, 1])))
    y_ax = _norm(np.cross(z_ax, x_ax))

    R = np.stack([x_ax, y_ax, z_ax], axis=-1)
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, -1] = origin

    return matrix

def create_direction_model(world_config, experiment_config: Dict[str, Any]):
    model = create_model(experiment_config)
    transforms_dict = create_transforms(experiment_config)
    model.to("cuda")
    run_name = experiment_config['logging']['run_name']
    checkpoint_path = experiment_config["training"].get("checkpoint_dir", f"checkpoints/{run_name}")
    checkpoint = torch.load(f"{checkpoint_path}/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.Sequential(
        Transform(partial(StereoSample.transform, transforms=transforms_dict['eval'])), # not turning things correctly
        model,
        Transform(lambda modeloutput: modeloutput.pred),
        Transform(box2robot),
        Transform(lambda out : out.detach().cpu().numpy().squeeze())
    )
    return model, transforms_dict

class KeyboardFlags:

    def __init__(self):
        self._step_flag = Event()
        self._quit_flag = Event()
        Thread(target=self._dispatch, daemon=True).start()

    def _dispatch(self):
        while not self._quit_flag.is_set():
            if input().strip() == "q":
                self._quit_flag.set()
            else:
                self._step_flag.set()

    def clear_step_flag(self):
        self._step_flag.clear()

    def quit_flag(self):
        return self._quit_flag.is_set()

    def step_flag(self):
        return self._step_flag.is_set()

def sample_in_box(corner1, corner2, n=1):
    low = np.minimum(corner1, corner2)
    high = np.maximum(corner1, corner2)
    return np.random.default_rng().uniform(low, high, size=(n, 3))

def get_image(camera) -> Image:
    return Image(camera.get_rgb(), camera.get_intrinsics_matrix(), camera.path)

def spawn_n_robots(config, parent: SceneObj, direction_policy, *, n: int, visualize_direction: bool):
    model_config = utils.load_config(config.model_config_path)
    #TODO: sample in box should be a sampling policy, a function passed in to get the robot positions
    sampled_poses = sample_in_box(config.near_corner, config.far_corner, n=n)
    direction_model, _ = create_direction_model(config, model_config)

    def create_robot(i, pose):
        name = f"robot_{i}"
        import robot as robo
        r =  robo.Robot(
            name,
            parent,
            pose,
            config.robot_init_rot,
            direction_model,
            direction_policy(),
            config.step_size,
            visualize_direction=visualize_direction,
        )
        return r

    robots = [create_robot(i, pose) for i, pose in enumerate(sampled_poses)]
    return robots

def hog(sim_app, keyboard_input):
    while sim_app.is_running() and not keyboard_input.step_flag() and not keyboard_input.quit_flag():
        sim_app.update()
    keyboard_input.clear_step_flag()

def prim_at_path(path):
    from omni.isaac.core.utils.stage import get_current_stage
    return get_current_stage().GetPrimAtPath(path)

def is_done(config):
    keyboard_input = KeyboardFlags() if config.debug else None
    countdown = (False for _ in range(config.sim_steps))
    def _is_done():
        return keyboard_input.quit_flag() if config.debug else next(countdown, True)
    return _is_done, keyboard_input


def videoify(steps: List[Step]):
    assert len(steps) > 0, "Steps list is empty"

    buffers = {}
    for step in steps:
        for renderable in step.renderables:
            buffers.setdefault(renderable.unique_name, []).append(renderable.rendered)

    for name, buffer in buffers.items():
        name = name.replace("/","_")
        imageio.mimsave(f"visual-servo-rollout/{name}.mp4", buffer, fps=20)

def physics_step(world):
    world.step(render=True)

@contextmanager
def render_step_hidden(sim_app, prims_to_hide: List[SceneObj]):
    [p.hide() for p in prims_to_hide]
    sim_app.update()
    try:
        yield
    finally:
        [p.unhide() for p in prims_to_hide]
        sim_app.update()

def render_step(sim_app):
    sim_app.update()
