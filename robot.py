import pdb
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple

from pxr import UsdGeom, Usd
import numpy as np
from isaacsim.core.api.objects import VisualCuboid
import torchvision.transforms.v2 as v2


from datagen2_isaacsim.isaac_utils import create_empty, set_transform
from hardwares import ZedMini
from rollout_datastructs import Observation, ObsAction, Action, Image, SceneObj
import transform_utils
from datastructs import StereoSample
import rollout_utils as ru
import transform_utils as tu


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

    def draw(self, debug_draw_interface):
        end = (self.origin + self.direction)
        debug_draw_interface.draw_lines([self.origin], [end], [self.color], [self.thickness])

@dataclass
class PointSpec:
    origin: np.ndarray
    color: Tuple[float, float, float, float] = (1., 0., 0., 1.)
    thickness: float = 3.0

    def draw(self, debug_draw_interface):
        debug_draw_interface.draw_points([self.origin], [self.color], [self.thickness])


class DebugTarget(SceneObj):
    def __init__(self, name, parent, color, thickness):
        self.path = f"{parent.path}/{name}"
        super().__init__(self.path, None)
        self.name = name
        self.parent = parent
        self.color = color
        self.thickness = thickness
        self.translation = (0, 0, 0)
        self.rotation = (0, 0, 0)
        self.transform(translation=self.translation, rotation=self.rotation)

    def transform(self, translation=None, rotation=None):
        self.translation = translation
        self.rotation = rotation

        parent2world = ru.prim_local2world(self.parent)
        self2parent = transform_utils.create_se3(translation=self.translation, active_euler=self.rotation)
        self2world = self2parent @ parent2world
        pose_world = transform_utils.get_translation(self2world)

        DebugRegistry.get().draw(
                self.name, PointSpec(pose_world, self.color, self.thickness)
        )

    def hide(self):
        DebugRegistry.get().hide(self.name)

    def unhide(self):
        self.transform(self.translation, self.rotation)

class DebugRegistry:
    """Global registry for debug arrows. Coordinates the singleton debug_draw interface."""
    _instance = None

    def __init__(self):
        self._db_objs: Dict[str, ArrowSpec | PointSpec] = {}
        from isaacsim.util.debug_draw import _debug_draw
        self._draw_iface = _debug_draw.acquire_debug_draw_interface()

    @classmethod
    def get(cls) -> "DebugRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def draw(self, key: str, spec: ArrowSpec | PointSpec):
        self._db_objs[key] = spec
        self._redraw_all()

    def hide(self, key: str):
        self._db_objs.pop(key, None)
        self._redraw_all()

    def _redraw_all(self):
        self._draw_iface.clear_lines()
        self._draw_iface.clear_points()
        [spec.draw(self._draw_iface) for spec in self._db_objs.values()]

# -----------------------------------------------------------------------------
# Robot
# -----------------------------------------------------------------------------
class ChainedPolicy:
    def __init__(self, *policy_objects):
        self._policies = policy_objects

    def __call__(self, direction):
        for p in self._policies:
            direction = p(direction)

        return direction

class ChainedPolicyFactory:
    def __init__(self):
        self._policy_factories = []

    def append(self, *factories):
        self._policy_factories += factories

    def __call__(self):
        return ChainedPolicy(*(factory() for factory in self._policy_factories))

class IdentityDirectionPolicy:
    def __call__(self, direction):
        return direction

class MovingAvgDirectionPolicy:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __call__(self, direction: np.ndarray):
        self.buffer.append(direction)
        return np.array(self.buffer).mean(axis=0)

class TransformDirectionPolicy:
    def __init__(self, translation, rotation):
        self._translation = translation
        self._rotation = rotation

    def __call__(self, direction):

        if self._translation:
            direction = direction + self._translation
        if self._rotation:
            direction = np.dot(self._rotation, direction)

        return direction

class Robot(SceneObj):

    def __init__(self, name, parent: SceneObj, init_translation, init_rotation, offset_model, direction_policy, step_size, *, visualize_direction: bool):
        self.path = f"{parent.path}/{name}"
        self.empty = create_empty(name, parent.path)
        super().__init__(self.path, self.empty)
        self.name = name
        self.next_direction_model = offset_model
        self.step_size = step_size
        self.transform(init_translation, init_rotation)

        self.last_direction = None

        self.camera = (
                ZedMini("camera", parent_path=self.path, frequency=-1)
                .transform(rotation=(0., 0., -90.))
        )

        self.direction_policy = direction_policy
        self.visualize_direction = visualize_direction
        self.body = VisualCuboid(
            prim_path=f"{self.path}/body",
            name="camera_body",
            size=0.025,
            color=np.array([0, 255, 0]),
        )

    def hide(self):
        super().hide()
        if self.visualize_direction:
            DebugRegistry.get().hide(self.name)

    def unhide(self):
        super().unhide()


def get_obs(robot) -> Observation:
    left_Image = ru.get_image(robot.camera.left_camera)
    left_pose = ru.prim_local2world(robot.camera.left_camera)
    # equivalent:
    # left_translation, left_rotn = robot.camera.left_camera.get_world_pose(camera_axes="usd")
    # from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
    # R = quat_to_rot_matrix(left_rotn) # assuming this is an active rotation: thus this is the body2world
    # left_pose = np.eye(4)
    # left_pose[:3, :3] = R
    # left_pose[:3, -1] = left_translation
    # pdb.set_trace()
    # left pose should be a left2world
    right_Image = ru.get_image(robot.camera.right_camera)
    right_pose = ru.prim_local2world(robot.camera.right_camera)

    depth = robot.camera.get_depth()

    return Observation(
            robot.name,
            ru.prim_local2world(robot),
            left_Image,
            right_Image,
            left_pose,
            right_pose,
            depth,
    )

def get_model_device(model):
    for p in model.parameters():
        return p.device

    for b in model.buffers():
        return b.device

    return "cpu"

def next_direction(robot, inpt: StereoSample) -> np.ndarray: 
    new_direction = robot.next_direction_model(inpt) 
    return robot.direction_policy(new_direction)

def move(action, robot):
    cur_pose = ru.prim_local2parent(robot)
    action = transform_utils.rotate(action, cur_pose)
    robot_new_pose = transform_utils.add_translation(action, cur_pose)
    robot.transform(
        translation=transform_utils.get_translation(robot_new_pose),
    )

def action_loop_once(robot):
    device = get_model_device(robot.next_direction_model)
    obs = get_obs(robot)
    stereo_sample = obs.stereo_sample().transform(v2.ToImage()).move_to(device)
    dir = next_direction(robot, stereo_sample)

    if robot.visualize_direction:
        robot2world = ru.prim_local2world(robot)
        robot_origin = transform_utils.get_translation(robot2world)
        DebugRegistry.get().draw(
            robot.name, 
            ArrowSpec(
                origin=robot_origin,
                direction=transform_utils.rotate(dir, robot2world)
            )
        )
    step = transform_utils.resize_norm(dir, robot.step_size)
    move(step, robot)
    robot_pose = ru.prim_local2world(robot)
    left_coords = robot.camera.left_camera.get_image_coords_from_world_points(tu.transform(dir, robot_pose)[None, ...])
    return ObsAction(obs, Action(dir, left_coords))

def set_direction_model(robot, direction_model):
    robot.next_direction_model = direction_model
