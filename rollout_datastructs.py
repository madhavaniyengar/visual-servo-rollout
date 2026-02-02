import pdb

from dataclasses import dataclass
from contextlib import contextmanager
from typing import List

import numpy as np

from datastructs import StereoSample

class PrimObj:
    def __init__(self, path, prim):

        self.path = path
        self.prim = prim

    def hide(self):
        from omni.isaac.core.utils.prims import set_prim_visibility
        set_prim_visibility(self.prim, False)

    def unhide(self):
        from omni.isaac.core.utils.prims import set_prim_visibility
        set_prim_visibility(self.prim, True)

class ExternalCamera(PrimObj):
    def __init__(self, position, orientation, prim_path="/World/external_camera",
                 resolution=(1920, 1080), frequency=20,
                 focal_length=1.8, clipping_range=(0.1, 100.0)):
        from isaacsim.sensors.camera import Camera
        
        camera = Camera(prim_path=prim_path, frequency=frequency, resolution=resolution)
        camera.initialize()
        camera.set_clipping_range(*clipping_range)
        camera.set_local_pose(translation=position, orientation=orientation, camera_axes="usd")
        camera.set_focal_length(focal_length)
        
        super().__init__(path=prim_path, prim=camera.prim)
        self._camera = camera
    
    def __getattr__(self, name):
        return getattr(self._camera, name)
    
class Empty(PrimObj):
    def __init__(self, parent: PrimObj, name: str):
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
