import os
from dataclasses import dataclass
import pdb

import numpy as np
import cv2
import tyro
import imageio
from isaacsim import SimulationApp

import transform_utils as tu
from rollout_datastructs import Image, Step
_debug_draw = None

# class Arrow:
#     def __init__(self, origin: np.ndarray, direction: np.ndarray, length: float):
#         self.origin = np.ndarray
#         self.direction = direction
#         self.length = length
#         self.debug_line = _debug_draw.acquire_debug_draw_interface()
#
#         self.debug_line.clear_lines()

class IsaacSimWorld:
    def __init__(self, stage, world, sim_app, external_camera, robot):
        self.stage = stage
        self.sim_app = sim_app
        self.world = world
        self.external_camera = external_camera
        # self.arrow = arrow
        self.robot = robot

@dataclass 
class WorldConfig:
    scene_path : str ='visual-servo-rollout/output_scene.usdz'
    e_cam_init_pos: tuple[float, float, float] = (0.15, -0.75, 0.35)
    e_cam_init_rot: tuple[float, float, float] = (70, 0, 0)

    robot_init_pose: tuple[float, float, float] = (0.15, -0.4, 0.15) 
    robot_init_rot: tuple[float, float, float] = (0., 0., 0.)

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
    external_camera.set_world_pose(
        position=config.e_cam_init_pos,
        orientation=rot_utils_np.euler_angles_to_quats(config.e_cam_init_rot, degrees=True),
        camera_axes="usd"
    )
    my_world = World(stage_units_in_meters=1.0)
    my_world.reset()
    robot = robo.Robot("robot", "/World", config.robot_init_pose, config.robot_init_rot, lambda _ : np.array([0., 0.05, 0.]))
    # arrow = Arrow(config.robot_init_pose, [1.0, 0., 0.], 1.)

    return IsaacSimWorld(
        stage, my_world, simulation_app, external_camera, robot#, arrow
    )

def step_seriously(world):
    world.world.step(render=True)
    for _ in range(10):
        world.sim_app.update()

def get_image(camera) -> Image:
   return Image(camera.get_rgb(), camera.prim_path)

def step_sim(world) -> Step:
    obs_action = robo.action_loop_once(world.robot)
    # so, this should actually also move the body too
    # update_arrow(obs_action, world.arrow)
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

    world = setup_isaacsim(config)
    step_seriously(world)
    imgs = []
    for i in range(10):
        step = step_sim(world)
        imgs.append(step.images[0].image)
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
