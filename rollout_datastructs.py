import pdb

from dataclasses import dataclass
from contextlib import contextmanager
from typing import List

import numpy as np

from datastructs import StereoSample

# class Wireframe(SceneObj):
#     def __init__(self, name, parent, orientation=None, translation=None):
#         super().__init__(name, None)
#         self.parent = parent
#
#     def transform(self, orientation=None, translation=None):
#         parent_world_pose = self.parent.world_pose

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
class Image:
    image: np.ndarray
    path: str

@dataclass
class Observation:
    left: Image
    right: Image
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


@dataclass 
class Step:
    images: List[Image]

    @property
    def camera_paths(self):
        return (image.path for image in self.images)

@dataclass 
class ObsAction:
    obs: Observation
    action: Action
