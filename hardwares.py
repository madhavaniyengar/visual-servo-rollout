"""Hardware definitions for gripper and camera setup in Isaac Sim."""

import math
import pdb

import numpy as np
from dataclasses import dataclass
from pxr import Gf, UsdGeom
import omni.replicator.core as rep
from omni.isaac.core.utils.stage import get_current_stage
import isaacsim.core.utils.numpy.rotations as rot_utils_np
from isaacsim.sensors.camera import Camera
from datagen2_isaacsim.isaac_utils import create_empty, setup_camera, set_transform, setup_render_product

from rollout_datastructs import SceneObj, RolloutCamera
import rollout_utils as ru

class ZedMini(SceneObj):
    """Zed Mini stereo camera simulation using Isaac Sim Camera API.

    Specs:
    RGB:
        1920x1080, 1280x720, 720x404
        Aperture: f/2.0
        Focal Length: 2.8mm (0.11")
        Field of View: 102 deg (H) x 57 deg (V) x 118 deg (D) max.
        Baseline: 63 mm
    """
    def __init__(self, name, parent_path, frequency, rgb_width: int = 1920, rgb_height: int = 1080):
        self.name = name
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.baseline = 0.063  # 63 mm
        self.focal_length = 0.28  # cm
        self.sensor_width_mm = 5.23  # 1/3" sensor width to match 102 deg FOV

        # Create rig root
        prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)
        super().__init__(prim_path, self.prim)

        # Left RGB camera
        self.left_camera_path = f"{self.path}/{name}_left"
        self.left_camera = RolloutCamera(
            parent=self,
            name=f"{name}_left",
            resolution=(self.rgb_width, self.rgb_height),
            frequency=frequency
        )
        self.left_camera.set_local_pose(
            [-self.baseline * 0.5, 0, 0],
            rot_utils_np.euler_angles_to_quats([90, 0, 0], degrees=True),
            camera_axes="usd"
        )


        # Right RGB camera
        self.right_camera_path = f"{self.path}/{name}_right"
        self.right_camera = RolloutCamera(
            parent=self,
            name=f"{name}_right",
            resolution=(self.rgb_width, self.rgb_height),
            frequency=frequency
        )
        self.right_camera.set_local_pose(
            [self.baseline * 0.5, 0, 0],
            rot_utils_np.euler_angles_to_quats([90, 0, 0], degrees=True),
            camera_axes="usd"
        )

        # Set optical parameters on both cameras
        self._set_camera_optics()
        self.initialize()

    def _set_camera_transform(self, camera_path, x_offset):
        """Set local transform for a camera."""
        stage = get_current_stage()
        cam_prim = stage.GetPrimAtPath(camera_path)
        set_transform(
            cam_prim,
            translation=(x_offset, 0.0, 0.0),
            rotation=(90.0, 0.0, 0.0)
        )

    def _set_camera_optics(self):
        """Set focal length and aperture for both cameras to match Zed Mini specs."""
        self.left_camera.set_focal_length(self.focal_length)
        self.right_camera.set_focal_length(self.focal_length)
        self.left_camera.set_clipping_range(0.001, 10000.0)
        self.right_camera.set_clipping_range(0.001, 10000.0)

        # Set horizontal/vertical aperture via USD (Camera class doesn't expose this)
        stage = get_current_stage()
        aspect_ratio = self.rgb_width / self.rgb_height
        vertical_aperture = self.sensor_width_mm / aspect_ratio

        for camera_path in [self.left_camera_path, self.right_camera_path]:
            cam_geom = UsdGeom.Camera.Get(stage, camera_path)
            cam_geom.GetHorizontalApertureAttr().Set(self.sensor_width_mm)
            cam_geom.GetVerticalApertureAttr().Set(vertical_aperture)

    def initialize(self):
        """Initialize both cameras. Must be called after world.reset()."""
        self.left_camera.initialize()
        self.right_camera.initialize()

    def get_depth(self):
        return self.left_camera.get_depth()

    def get_left_rgb(self):
        """Get RGBA image from left camera."""
        return self.left_camera.get_current_frame(clone=True)["rgb"][..., :3]

    def get_right_rgb(self):
        """Get RGBA image from right camera."""
        return self.right_camera.get_current_frame(clone=True)["rgb"][..., :3]

    def get_left_frame(self):
        """Get full frame data from left camera."""
        return self.left_camera.get_current_frame()

    def get_right_frame(self):
        """Get full frame data from right camera."""
        return self.right_camera.get_current_frame()

