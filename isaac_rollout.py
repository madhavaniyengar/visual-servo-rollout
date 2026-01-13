import os
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
ru = None
from rollout_datastructs import Image, ObsAction, Step, PrimObj, ExternalCamera, SimWorld, Empty
import utils

class IsaacSimWorld:
    def __init__(self, stage, world, sim_app, external_camera, robots: List[PrimObj]):
        self.stage = stage
        self.sim_app = sim_app
        self.world = world

        self.external_camera = external_camera
        self.robots = robots

@dataclass
class Config:
    headless: bool = False
    model_config_path: str = "experiment_configs/best_rollout.yaml"
    scene_path : str ='visual-servo-rollout/output_scene.usdz'
    sim_steps: int = 100
    e_cam_init_pos: tuple[float, float, float] = (-0.4018, 0.05, 0.25)

    near_corner: tuple[float, float, float] = (0.35, -0.15, 0.05)
    far_corner: tuple[float, float, float] = (0.65, 0.15, 0.15)  # going to be defined in the box coordinate frame
    robot_init_rot: tuple[float, float, float] = (0., 0., 180.)
    n_robots: int = 10
    robot_first_direction_only: bool = False
    step_size: float = 0.001

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


def setup_isaacsim(config) -> IsaacSimWorld:
    simulation_app = SimulationApp({
        "headless": config.headless,
        "width": 1920,
        "height": 1080,
        "/app/asyncRendering": False,
        "/app/asyncRenderingLowLatency": False,
        "/omni/replicator/asyncRendering": False,
    })
    global omni, rot_utils_np, rot_utils
    global World, GroundPlane, VisualCuboid, VisualCylinder, VisualCone, Camera
    global Sdf, UsdLux, UsdGeom, Gf, Usd
    global set_camera_view, robo, ru
    global rep, set_prim_visibility
    global get_current_stage

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

    external_camera = ExternalCamera(
        position=config.e_cam_init_pos,
        orientation=np.asarray([0.56425, 0.50051, -0.43144, -0.49494]),
    )
    pallet = ru.pfind("LoadedPallet_0", stage)
    ru.pmodify(pallet, translation=(0., 0.4, 0.06))
    sim_world = SimWorld(stage_units_in_meters=1.0)
    grasp_frame = Empty(parent=pallet, name="grasp_frame")
    ru.pmodify(grasp_frame, rotation=(0, 0, -90))
    
    robots = ru.spawn_n_robots(
        config=config,
        parent=grasp_frame,
        sim_app=simulation_app,
        n=config.n_robots,
    )

    return IsaacSimWorld(
        stage, sim_world, simulation_app, external_camera, robots
    )

def step_sim(world, keyboard_input: Optional["KeyboardFlags"]) -> Step:
    ru.hog(world.sim_app, keyboard_input) if keyboard_input else None
    obs_actions = [robo.action_loop_once(r) for r in world.robots]
    images = list(map(lambda oa : oa.obs.left, obs_actions))
    external_image = ru.get_image(world.external_camera)
    ru.physics_step(world.world)
    ru.render_step(world.sim_app)

    return Step(list(itertools.chain(images, [external_image])))

def main(config):

    world = setup_isaacsim(config)

    is_done, keyboard_input = ru.is_done(config)
    steps = []
    while not is_done():
        step = step_sim(world, keyboard_input)
        steps.append(step)

    ru.videoify(steps)

    world.sim_app.close()

if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
