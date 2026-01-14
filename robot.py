import pdb

from pxr import UsdGeom, Usd
import numpy as np
from isaacsim.core.api.objects import VisualCuboid


from datagen2_isaacsim.isaac_utils import create_empty, set_transform
from hardwares import ZedMini
from rollout_datastructs import Observation, ObsAction, Action, Image
import transform_utils


class Robot:

    def __init__(self, name, parent_path, init_translation, init_rotation, offset_model):
        self.path = f"{parent_path}/{name}"
        self.empty = create_empty(name, parent_path)
        self.next_direction_model = offset_model
        set_transform(self.empty, init_translation, init_rotation) # when init_rotation is non identity, it might be problematic to change
        # the rotation at runtime... Might be time to re learn how usd works

        self.camera = ZedMini("camera", parent_path=self.path)
        set_transform(self.camera.prim, rotation=(0., 0., -90.))
        self.body = VisualCuboid(
            prim_path=f"{self.path}/body",
            name="camera_body",
            size=0.025, # xform rel to parent is identity
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

def prim_get_pose(prim):
    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(matrix).T

def get_pose(robot):
    return prim_get_pose(robot.empty)

def next_direction(robot, observation: Observation) -> np.ndarray: 
    stereo_sample = observation.stereo_sample()
    direction = robot.next_direction_model(stereo_sample) # in robot coordinate frame
    robot2world = get_pose(robot) # should be equivalent to prim_get_pose(robot.camera)
    direction = transform_utils.rotate(direction, robot2world)
    return direction

def move(action, robot):
    cur_pose = get_pose(robot)
    robot_new_pose = transform_utils.add_translation(action, cur_pose)
    set_transform(
            robot.empty,
            transform_utils.get_translation(robot_new_pose),
            transform_utils.get_euler(robot_new_pose)
    )
    # camera.set_world_pose(position=camera_pos, orientation=CAMERA_QUAT, camera_axes="usd")
    # this hopefuly is identical: nope it's not. Weird mangling when you use set_world_pose

def action_loop_once(robot):
    obs = get_obs(robot)
    dir = next_direction(robot, obs)
    move(dir, robot)
    return ObsAction(obs, Action(dir))





