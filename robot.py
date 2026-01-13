import pdb
from dataclasses import dataclass, field
from typing import Dict, Tuple

from pxr import UsdGeom, Usd
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
import torchvision.transforms.v2 as v2


from datagen2_isaacsim.isaac_utils import create_empty, set_transform
from hardwares import ZedMini
from rollout_datastructs import Observation, ObsAction, Action, Image, PrimObj
import transform_utils
from datastructs import StereoSample
import rollout_utils as ru


# -----------------------------------------------------------------------------
# Arrow Drawing (mechanism)
# -----------------------------------------------------------------------------

@dataclass
class ArrowSpec:
    """Specification for a single debug arrow."""
    origin: np.ndarray
    direction: np.ndarray
    color: Tuple[float, float, float, float] = (1., 0., 0., 1.)  # RGBA
    thickness: float = 3.0


class ArrowRegistry:
    """Global registry for debug arrows. Coordinates the singleton debug_draw interface."""
    _instance = None

    def __init__(self):
        self._arrows: Dict[str, ArrowSpec] = {}
        self._draw_iface = None
        self._carb = None

    @classmethod
    def get(cls) -> "ArrowRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_arrow(self, key: str, spec: ArrowSpec):
        self._arrows[key] = spec
        self._redraw_all()

    def remove_arrow(self, key: str):
        self._arrows.pop(key, None)
        self._redraw_all()

    def _redraw_all(self):
        # Lazy import to avoid import-time side effects
        if self._draw_iface is None:
            from isaacsim.util.debug_draw import _debug_draw
            import carb
            self._draw_iface = _debug_draw.acquire_debug_draw_interface()
            self._carb = carb

        self._draw_iface.clear_lines()
        for spec in self._arrows.values():
            start = self._carb.Float3(*spec.origin)
            end = self._carb.Float3(*(spec.origin + spec.direction))
            color = self._carb.ColorRgba(*spec.color)
            self._draw_iface.draw_lines([start], [end], [color], [spec.thickness])


# -----------------------------------------------------------------------------
# Robot
# -----------------------------------------------------------------------------

class Robot(PrimObj):

    def __init__(self, name, parent_path, init_translation, init_rotation, offset_model, sim_app, first_direction_only, step_size):
        self.path = f"{parent_path}/{name}"
        self.empty = create_empty(name, parent_path)
        super().__init__(self.path, self.empty)
        self.name = name
        self.next_direction_model = offset_model
        self.step_size = step_size
        ru.pmodify(self, init_translation, init_rotation)

        self.first_direction_only = first_direction_only
        self.last_direction = None

        camera = ZedMini("camera", parent_path=self.path)
        filtered_camera = ru.FilteredCamera(camera, sim_app)
        self.camera = filtered_camera
        ru.pmodify(self.camera, rotation=(0., 0., -90.))

        self.body = VisualCuboid(
            prim_path=f"{self.path}/body",
            name="camera_body",
            size=0.025,
            color=np.array([0, 255, 0]),
        )

def get_obs(robot) -> Observation:
    left = robot.camera.get_left_rgb()
    right = robot.camera.get_right_rgb()
    depth = robot.camera.get_depth()

    return Observation(
            Image(left, robot.camera.left_camera_path),
            Image(right, robot.camera.right_camera_path),
            depth,
    )

def get_model_device(model):
    for p in model.parameters():
        return p.device

    for b in model.buffers():
        return b.device

    return "cpu"

def next_direction(robot, inpt: StereoSample) -> np.ndarray: 
    if robot.first_direction_only:
        if robot.last_direction is None:
            robot.last_direction = robot.next_direction_model(inpt) # in robot coordinate frame
        return robot.last_direction

    return robot.next_direction_model(inpt)

def move(action, robot):
    cur_pose = ru.prim_local2parent(robot)
    action = transform_utils.rotate(action, cur_pose)
    robot_new_pose = transform_utils.add_translation(action, cur_pose)
    ru.pmodify(
        robot,
        translation=transform_utils.get_translation(robot_new_pose),
    )

def action_loop_once(robot):
    device = get_model_device(robot.next_direction_model)
    obs = get_obs(robot)
    dir = next_direction(
        robot,
        obs.stereo_sample().transform(v2.ToImage()).move_to(device)
    )
    # Update the robot's arrow in the global registry
    robot2world = ru.prim_local2world(robot)
    robot_origin = transform_utils.get_translation(robot2world)
    ArrowRegistry.get().set_arrow(
        robot.name, 
        ArrowSpec(
            origin=robot_origin,
            direction=transform_utils.rotate(dir, robot2world)
        )
    )
    step = transform_utils.resize_norm(dir, robot.step_size)
    move(step, robot)
    return ObsAction(obs, Action(dir))

def set_direction_model(robot, direction_model):
    robot.next_direction_model = direction_model
