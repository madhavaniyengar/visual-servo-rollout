import pdb

from abc import ABC, abstractmethod
import cv2
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List

import numpy as np

from datastructs import StereoSample
import transform_utils as tu


# class Wireframe(SceneObj):
#     def __init__(self, name, parent, orientation=None, translation=None):
#         super().__init__(name, None)
#         self.parent = parent
#
#     def transform(self, orientation=None, translation=None):
#         parent_world_pose = self.parent.world_pose

class Renderable(ABC):
    @property
    @abstractmethod
    def unique_name(self) -> str:
        ...

    @property
    @abstractmethod
    def rendered(self):
        ...

class SceneObj:
    def __init__(self, path, prim):
        self.path = path
        self.prim = prim

    def transform(self, translation=None, rotation=None):
        from datagen2_isaacsim.isaac_utils import create_empty, set_transform
        set_transform(self.prim, translation, rotation)
        return self


    def hide(self):
        from omni.isaac.core.utils.prims import set_prim_visibility
        set_prim_visibility(self.prim, False)

    def unhide(self):
        from omni.isaac.core.utils.prims import set_prim_visibility
        set_prim_visibility(self.prim, True)

class RolloutCamera(SceneObj):
    def __init__(self, parent: SceneObj, name, 
                 resolution=(1920, 1080), frequency=20,
                 focal_length=1.8, clipping_range=(0.1, 100.0)):
        from isaacsim.sensors.camera import Camera
        
        prim_path = f"{parent.path}/{name}"
        camera = Camera(prim_path=prim_path, frequency=frequency, resolution=resolution)
        camera.initialize()
        camera.set_clipping_range(*clipping_range)
        camera.set_focal_length(focal_length)
        
        super().__init__(path=prim_path, prim=camera.prim)
        self._camera = camera

    def transform(self, translation=None, rotation=None):
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        orientation = euler_angles_to_quat(rotation, degrees=True, extrinsic=True) if rotation else None
        # pdb.set_trace() # make sure #np.asarray([0.56425, 0.50051, -0.43144, -0.49494]) you get this for the external camera
        self._camera.set_local_pose(translation=translation, orientation=orientation, camera_axes="usd")
        return self

    def get_rgb(self):
        return self._camera.get_current_frame(clone=True)["rgb"][..., :3]
    
    def __getattr__(self, name):
        return getattr(self._camera, name)
    
class Empty(SceneObj):
    def __init__(self, parent: SceneObj, name: str):
        from datagen2_isaacsim.isaac_utils import create_empty
        super().__init__(f"{parent.path}/{name}", create_empty(name, parent.path))
        

class SimWorld:
    def __init__(self, stage_units_in_meters=1.0):
        from isaacsim.core.api import World
        
        self._world = World(stage_units_in_meters=stage_units_in_meters)
        self._world.reset()
    
    def __getattr__(self, name):
        return getattr(self._world, name)

@dataclass
class Image(Renderable):
    image: np.ndarray
    intrinsics: np.ndarray
    path: str

    @property
    def unique_name(self):
        return self.path

    @property
    def rendered(self):
        return self.image

@dataclass
class Observation:
    robot_name: str
    robot_pose: np.ndarray
    left: Image
    right: Image
    left_pose: np.ndarray
    right_pose: np.ndarray
    depth: np.ndarray

    def flatten(self):
        return [self.left.image, self.right.image, self.depth], dict(left_path=self.left.path, right_path=self.right.path)

    def stereo_sample(self):
        return StereoSample(
            self.left.image,
            self.right.image,
            self.depth,
            None,
            None,
            None,
            None
        )

@dataclass 
class Action:
    direction: np.ndarray
    left_coords: np.ndarray


@dataclass 
class Step:
    renderables: List[Renderable]

@dataclass
class ObsAction(Renderable):
    obs: Observation
    action: Action

    @property
    def unique_name(self):
        return self.obs.robot_name

    @property
    def rendered(self):
        # left_K = self.obs.left.intrinsics

        # baseline2world = self.obs.robot_pose
        # left2world = self.obs.left_pose
        # baseline2left = baseline2world @ tu.se3_inverse(left2world)
        # left2baseline = np.eye(4)
        # left2baseline[:3, :3] = np.array(
        #     [
        #         [0, 0, -1],
        #         [-1, 0, 0],
        #         [0, 1, 0]
        #     ]
        # )
        # left2baseline[:3, -1] = np.array([0, 0.063/2, 0])

        # baseline2left = tu.se3_inverse(left2baseline)
        # image_coord = np.dot(left_K, tu.transform(predicted_vector, baseline2left))
        # image_coord = image_coord[:-1] / image_coord[-1]
        # image_coord = tuple(map(int, image_coord))
        left_image = np.ascontiguousarray(self.obs.left.image)
        coords = tuple(map(int, self.action.left_coords.squeeze()))
        cv2.circle(left_image, coords, radius=6, color=(255, 0, 0), thickness=5)
        return left_image
