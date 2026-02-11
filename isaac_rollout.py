import os
from functools import partial
import itertools
import threading
from dataclasses import dataclass, fields
from pathlib import Path
import pdb
from typing import Callable, Optional, List

import yaml

import numpy as np
import cv2
import tyro
import imageio
from isaacsim import SimulationApp

import transform_utils as tu

from rollout_datastructs import Image, ObsAction, Step, SceneObj, RolloutCamera, SimWorld, Empty
import utils

class IsaacSimWorld:
    def __init__(self, stage, world, sim_app, config, obj_registry):
        self.stage = stage
        self.sim_app = sim_app
        self.world = world

        self.config = config
        self.obj_registry = obj_registry

@dataclass
class Config:
    headless: bool = False
    model_config_path: str = "experiment_configs/best_rollout.yaml"
    scene_path : str ='visual-servo-rollout/output_scene.usdz'
    sim_steps: int = 100
    e_cam_init_pos: tuple[float, float, float] = (-0.5018, 0.00, 0.25)
    center_box_predictions: bool = False

    near_corner: tuple[float, float, float] = (0.55, -0.15, 0.05)
    far_corner: tuple[float, float, float] = (0.65, 0.15, 0.15)  # going to be defined in the box coordinate frame
    robot_init_rot: tuple[float, float, float] = (0., 0., 180.)
    n_robots: int = 10
    step_size: float = 0.001
    direction_use_moving_avg: bool = True
    direction_moving_avg_buffer_len: int = 20

    debug: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        field_names = {f.name for f in fields(cls)}
        provided_fields = set(data.keys())
        missing_fields = field_names - provided_fields

        if missing_fields:
            raise ValueError(
                f"Missing required fields in YAML config: {sorted(missing_fields)}. "
                f"Provided fields: {sorted(provided_fields)}. "
                f"Required fields: {sorted(field_names)}."
            )

        return cls(**data)

class ObjRegistry:

    def __init__(self):
        self._scene_objs: List[SceneObj] = []

    def register(self, *scene_objs):
        self._scene_objs.extend(scene_objs)

    def getall(self, *classes):
        return [obj for obj in self._scene_objs if isinstance(obj, classes)]

def setup_isaacsim(config) -> IsaacSimWorld:
    simulation_app = SimulationApp({
        "headless": config.headless,
        "width": 1920,
        "height": 1080,
    }, experience=f"{os.path.expanduser('~')}/isaacsim/apps/isaacsim.exp.base.zero_delay.kit")
    global omni, rot_utils_np, rot_utils
    global World, GroundPlane, VisualCuboid, VisualCylinder, VisualCone, Camera
    global Sdf, UsdLux, UsdGeom, Gf, Usd
    global set_camera_view, robo, ru
    global rep, set_prim_visibility
    global get_current_stage
    global euler_angles_to_quat

    import omni.usd
    import isaacsim.core.utils.numpy.rotations as rot_utils_np
    import isaacsim.core.utils.rotations as rot_utils
    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.api.objects import VisualCuboid, VisualCylinder, VisualCone
    from isaacsim.sensors.camera import Camera
    from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
    from isaacsim.core.utils.viewports import set_camera_view
    import omni.replicator.core as rep
    from omni.isaac.core.utils.prims import set_prim_visibility
    from omni.isaac.core.utils.stage import get_current_stage
    from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix

    import robot as robo
    import rollout_utils as ru

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    world_prim = stage.DefinePrim("/World", "Xform")
    world_prim.GetReferences().AddReference(config.scene_path, "/World")
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(1000)
    domeLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    domeLight.CreateIntensityAttr(500)

    obj_registry = ObjRegistry()

    external_camera = (
            RolloutCamera(
                parent=SceneObj("/World", world_prim),
                name="external_camera"
            )
           .transform(
                translation=config.e_cam_init_pos,
                rotation=(82.72115958, 0.49073557, -82.08012841) #np.asarray([0.56425, 0.50051, -0.43144, -0.49494]),
            )

    )
    obj_registry.register(external_camera)

    pallet = (
            ru.pfind("LoadedPallet_0", stage)
            .transform(translation=(0., 0.4, 0.042), rotation=(0., -0.5, 0))
    )

    grasp_frame = (
            Empty(parent=pallet, name="grasp_frame")
            .transform(translation=(0, -0.10/2, 0), rotation=(0, 0, -90))
    )

    policy_factory = robo.ChainedPolicyFactory()
    if config.direction_use_moving_avg:
        policy_factory.append(partial(robo.MovingAvgDirectionPolicy, maxlen=config.direction_moving_avg_buffer_len))
    else:
        policy_factory.append(robo.IdentityDirectionPolicy)

    if config.center_box_predictions:
        policy_factory.append(partial(robo.TransformDirectionPolicy, translation=(0, 0, 0.5 * pallet.height), rotation=None))


    robots = ru.spawn_n_robots(
        config=config,
        parent=grasp_frame,
        direction_policy_factory=policy_factory,
        n=config.n_robots,
        visualize_direction=True
        #TODO: you're basically keeping the arrow registry internal to the robot
    )
    obj_registry.register(*robots)

    fpv_vis = [
        RolloutCamera(parent=robot, name=f"vis_{robot.name}_camera").transform(rotation=[90, 0, -90])
        for robot in robots
    ]
    obj_registry.register(*fpv_vis)

    debug_target = (
            robo.DebugTarget(name="gt target", parent=grasp_frame, color=(0., 1., 0., 1.), thickness=50)
            .transform(translation=(0, 0, pallet.height/2))
    )
    obj_registry.register(debug_target)

    sim_world = SimWorld(stage_units_in_meters=1.0)

    return IsaacSimWorld(
        stage, sim_world, simulation_app, config, obj_registry
    )

def step_sim(world, keyboard_input: Optional["KeyboardFlags"]) -> Step:
    ru.hog(world.sim_app, keyboard_input) if keyboard_input else None

    ru.physics_step(world.world)
    ru.render_step(world.sim_app)
    vis_images = [ru.get_image(camera) for camera in world.obj_registry.getall(RolloutCamera)]

    with ru.render_step_hidden(world.sim_app, prims_to_hide=world.obj_registry.getall(robo.Robot, robo.DebugTarget)):
        robots = world.obj_registry.getall(robo.Robot)
        obs_actions = [robo.action_loop_once(r) for r in robots]

    return Step(vis_images + obs_actions)

def main(config):

    world = setup_isaacsim(config)

    is_done, keyboard_input = ru.is_done(config)
    steps = []
    while not is_done():
        step_ = step_sim(world, keyboard_input)
        steps.append(step_)

    ru.videoify(steps)

    world.sim_app.close()

if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
