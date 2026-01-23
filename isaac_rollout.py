import os
import threading
from dataclasses import dataclass, fields
from pathlib import Path
import pdb
from typing import Callable, Optional

import yaml

import numpy as np
import cv2
import tyro
import imageio
from isaacsim import SimulationApp
_debug_draw = None # actually an import, see setup_isaacsim()

import transform_utils as tu
from rollout_datastructs import Image, ObsAction, Step
import rollout_utils as ru
import utils


class Arrow:
    def __init__(self):
        self.debug_line = _debug_draw.acquire_debug_draw_interface()
        self.debug_line.clear_lines()

class IsaacSimWorld:
    def __init__(self, stage, world, sim_app, external_camera, robot, arrow):
        self.stage = stage
        self.sim_app = sim_app
        self.world = world
        self.external_camera = external_camera
        self.arrow = arrow
        self.robot = robot

@dataclass
class WorldConfig:
    model_config_path: str = "experiment_configs/onebox.yaml"
    scene_path : str ='visual-servo-rollout/output_scene.usdz'
    sim_steps: int = 50
    e_cam_init_pos: tuple[float, float, float] = (-0.4018, -0.15, 0.15)

    robot_init_pose: tuple[float, float, float] = (0.15, -0.2, 0.15)
    robot_init_rot: tuple[float, float, float] = (0., 0., 90.)

    debug: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "WorldConfig":
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
        "headless": True,
        "width": 1920,
        "height": 1080,
    })
    global omni, rot_utils_np, rot_utils
    global World, GroundPlane, VisualCuboid, VisualCylinder, VisualCone, Camera
    global Sdf, UsdLux, UsdGeom, Gf, Usd
    global set_camera_view, _debug_draw, carb, robo
    global rep

    import omni.usd
    import isaacsim.core.utils.numpy.rotations as rot_utils_np
    import isaacsim.core.utils.rotations as rot_utils
    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.api.objects import VisualCuboid, VisualCylinder, VisualCone
    from isaacsim.sensors.camera import Camera
    from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.util.debug_draw import _debug_draw
    import omni.replicator.core as rep
    import carb

    import robot as robo

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.DefinePrim("/World", "Xform")
    assert os.path.exists(config.scene_path), f"{config.scene_path} does not exist!"
    world_prim.GetReferences().AddReference(config.scene_path, "/World")
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(1000)
    domeLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    domeLight.CreateIntensityAttr(500)
    # green_cylinder = VisualCylinder(
    #     prim_path="/World/green_cylinder",
    #     name="green_cylinder",
    #     translation=np.array([0.1, 0.055, 0.05]),
    #     orientation=cylinder_quat,
    #     scale=np.array([0.01, 0.01, 0.001]),
    #     color=np.array([0, 255, 0]),  # Green color
    # )
    external_camera = Camera(
        prim_path="/World/external_camera",
        frequency=20,
        resolution=(1920, 1080),
    )
    external_camera.initialize()
    external_camera.set_clipping_range(0.1, 100.0)
    external_camera.set_local_pose(
        translation=config.e_cam_init_pos,
        orientation=np.asarray([0.56425, 0.50051, -0.43144, -0.49494]), # rot_utils_np.euler_angles_to_quats(config.e_cam_init_rot, degrees=True),
        camera_axes="usd"
    )
    external_camera.set_focal_length(1.8) # cm
    my_world = World(stage_units_in_meters=1.0)
    my_world.reset()
    robot = robo.Robot("robot", "/World", config.robot_init_pose, config.robot_init_rot, lambda _ : np.array([0.05, 0., 0.]))
    arrow = Arrow()

    return IsaacSimWorld(
        stage, my_world, simulation_app, external_camera, robot, arrow
    )

def step_seriously(world):
    world.world.step(render=True)
    for _ in range(10):
        world.sim_app.update()

def get_image(camera) -> Image:
   return Image(camera.get_rgb(), camera.prim_path)

def hog(sim_app, keyboard_input):
    while sim_app.is_running() and not keyboard_input.step_flag() and not keyboard_input.quit_flag():
        sim_app.update()
    keyboard_input.clear_step_flag()

def update_arrow(direction: np.ndarray, origin: np.ndarray, arrow):
    arrow.debug_line.clear_lines()
    arrow_end = origin + direction
    start = carb.Float3(*origin)
    endpoint = carb.Float3(*arrow_end)
    arrow_color = carb.ColorRgba(1., 0., 0., 1.)
    arrow_thickness = 3.
    arrow.debug_line.draw_lines(
        [start],  # List of carb.Float3
        [endpoint],    # List of carb.Float3
        [arrow_color],  # List of carb.ColorRgba
        [arrow_thickness]  # List of float
    )


def step_sim(world, keyboard_input: Optional[ru.KeyboardFlags]) -> Step:
    hog(world.sim_app, keyboard_input) if keyboard_input else None
    obs_action = robo.action_loop_once(world.robot)
    update_arrow(obs_action.action.direction, tu.get_translation(robo.get_pose(world.robot)), world.arrow)
    external_image = get_image(world.external_camera)
    step_seriously(world)

    return Step([obs_action.obs.left, external_image])

# def videoify(steps: List[Step]):
#     def _videoify(buffers, Step):
#         for i, image in enumerate(Step.images):
#             buffers[i].append(image) # now assuming that external camera also is a stereo camera or only uses the left_img
#
#     num_images = len(steps[0].images)
#     buffers = [[] for _ in range(num_images)]
#     prim_paths = [im.path for im in steps[0].images]
#     for step in steps:
#         _videoify(buffers, step)
#
#     for buf, prim path in zip(buffers, prim_paths):
#         turn buf into an mp4 using the prim path as the path

def main(config):

    model_config = utils.load_config(config.model_config_path)
    world = setup_isaacsim(config)
    direction_model, _ = ru.create_direction_model(config, model_config)
    robo.set_direction_model(world.robot, direction_model)
    step_seriously(world)
    imgs = []
    keyboard_input = ru.KeyboardFlags() if config.debug else None
    countdown = (True for _ in range(config.sim_steps))
    while (not keyboard_input.quit_flag()) if config.debug else next(countdown, False):
        step = step_sim(world, keyboard_input)
        imgs.append(step.images[1].image)
    imageio.mimwrite("test_video.mp4", imgs, fps=2)


    # try:
    #     while world.sim_app.is_running():
    #         # Update the simulation app to pro
    #         world.sim_app.update()
    #         # Optionally step the world if you
    #     # my_world.step(render=True)
    # except KeyboardInterrupt:
    #     print("\nShutting down simulation...")
    world.sim_app.close()
    # all_obs = []
    # for step in range(num_steps):
    #     obs_list = step_sim(world)
    #     all_obs.append(obs_list)

    world.sim_app.close()

if __name__ == "__main__":
    config = tyro.cli(WorldConfig)
    main(config)
