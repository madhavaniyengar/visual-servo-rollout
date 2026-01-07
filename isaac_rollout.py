from dataclasses import dataclass
import pdb

import cv2
import tyro
from isaacsim import SimulationApp

import transform_utils as tu
# @dataclass 
# class Image:
#     image: np.ndarray
#     path: str

# @dataclass 
# class Step:
#     images: List[Image]

# @dataclass
# class Observation:
#     left: Image
#     right: Image
#     depth: np.ndarray
#
#     def stereo_sample(self):
#         return StereoSample(
#             left.image,
#             right.image,
#             depth,
#             None,
#             None
#         )

# @dataclass 
# class Action:
#     direction: np.ndarray
#     current_pose: np.ndarray

# @dataclass 
# class ObsAction:
#     obs: Observation
#     action: Action

# class Arrow:
#     def __init__(self, origin: np.ndarray, direction: np.ndarray, length: float):
#         self.origin = np.ndarray
#         self.direction = direction
#         self.length = length
#         self.debug_line = _debug_draw.acquire_debug_draw_interface()
#
#         self.debug_line.clear_lines()

class IsaacSimWorld:
    def __init__(self, stage, world, sim_app, external_camera):
        self.stage = stage
        self.sim_app = sim_app
        self.world = world
        self.external_camera = external_camera
        # self.arrow = arrow
        # self.robot = robot

@dataclass 
class WorldConfig:
    scene_path : str ='output_scene.usdz'
    e_cam_init_pos: tuple[float, float, float] = (0.15, -0.75, 0.35)
    e_cam_init_rot: tuple[float, float, float] = (70, 0, 0)
    # robot init position: List[float] = 
    # robot init rotxyz : List[float] = 

def setup_isaacsim(config) -> IsaacSimWorld:
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1920,
        "height": 1080,
    })
    global omni, rot_utils_np, rot_utils, World, GroundPlane, VisualCuboid, VisualCylinder, VisualCone, Camera, Sdf, UsdLux, UsdGeom, Gf, Usd, set_camera_view, _debug_draw, carb

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
    import carb

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.DefinePrim("/World", "Xform")
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
    # robot = Robot()
    # arrow = Arrow()
    my_world.reset()

    return IsaacSimWorld(
        stage, my_world, simulation_app, external_camera
    )

def step_seriously(world):
    world.world.step(render=True)
    for _ in range(10):
        world.sim_app.update()

# def step_sim(isaacsimworld) -> Step
#     obs_action = action_loop_once(world.robot) # cute idea later: use getattr to do something like: action_loop_once(world.robot, "get_obs", 
#                                         # "next_direction", "move" and have as a protocol that each of these functions' outputs can be piped to each other)
#                                         # I could add decorators to pass through certain parameters (leave them untouched) and to return them
#     update_arrow(obs_action, world.arrow)
#     external_image = get_image(world.external_camera)
#     step_seriously()
#
#     return Step([obs_action.observation.left, external_image])

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
    image = world.external_camera.get_rgb()
    cv2.imwrite("test_image.png", image)
    world.sim_app.close()

    # all_obs = []
    # for step in range(num_steps):
    #     obs_list = step_sim(world)
    #     all_obs.append(obs_list)
    #
    # videoify(all_obs)


if __name__ == "__main__":
    config = tyro.cli(WorldConfig)
    main(config)
    # 2026-01-09T23:22:35Z [18,800ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World>: Could not open asset @visual-servo-rollout/output_scene.usdz@ for reference introduced by @anon:0x239a36d0:World1.usd@</World>. (recomposing stage on stage @anon:0x239a36d0:World1.usd@ <0x12aa0b60>)

