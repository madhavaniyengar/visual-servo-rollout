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

from rollout_datastructs import PrimObj


@dataclass
class OrbbecGemini2Args:
    """Configuration for OrbbecGemini2 cameras."""
    rgb_width: int = 1920
    rgb_height: int = 1080
    rgb_depth: int = 8
    rgb_format: str = 'PNG'
    depth_width: int = 1280
    depth_height: int = 800


class OrbbecGemini2:
    """Orbbec Gemini 2 stereo camera simulation.
    
    Specs:
    RGB:
        1920x1080, 1280x720, 640x480, 640x360
        Fov: 86°x55° (16:9) 63°x50° (4:3)
        Fps: 5, 10, 15, 30, 60
    
    Depth:
        1280x800, 640x400, 320x200
        91°x66°
        Fps: 5, 10, 15, 30, (60 binned only)
        Depth: 0.20 m -- 2.5 m
    
    Layout (50mm baseline):
        |-----------------------| 50.00 mm +/- 0.1 mm (?)
        |      |----|              8.50 mm
        |           |-----|       11.00 mm
        |           |-----------| 25.00 mm
      ( + )   (+) ( + ) ( + ) ( + )
       IR     LDP  LDM   RGB   IR
      [Right]                [Left]
    """
    
    def __init__(self, name, parent_path, args):
        """Initialize the Orbbec Gemini 2 camera rig.
        
        Args:
            name: Name of the camera rig
            parent_path: Parent prim path
            args: OrbbecGemini2Args configuration
        """
        self.name = name
        self.args = args
        stage = get_current_stage()
        
        # Create empty transform for camera rig
        self.prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)
        
        # Create IR cameras (stereo pair)
        self.ir_left_camera_path = f"{self.prim_path}/{name}_ir_left"
        self.ir_left_camera = self._create_ir_camera(
            f"{name}_ir_left", 
            self.ir_left_camera_path,
            args
        )
        # Position: -0.025m (25mm to the left)
        set_transform(
            self.ir_left_camera,
            translation=(-0.025, 0.0, 0.0),
            rotation=(180, 0, 0)  # Flipped
        )
        
        self.ir_right_camera_path = f"{self.prim_path}/{name}_ir_right"
        self.ir_right_camera = self._create_ir_camera(
            f"{name}_ir_right",
            self.ir_right_camera_path,
            args
        )
        # Position: +0.025m (25mm to the right)
        set_transform(
            self.ir_right_camera,
            translation=(0.025, 0.0, 0.0),
            rotation=(180, 0, 0)  # Flipped
        )
        
        # Create RGB camera
        self.rgb_camera_path = f"{self.prim_path}/{name}_rgb"
        self.rgb_camera = self._create_rgb_camera(
            f"{name}_rgb",
            self.rgb_camera_path,
            args
        )
        # Position: +0.011m (11mm offset from center)
        set_transform(
            self.rgb_camera,
            translation=(0.011, 0.0, 0.0),
            rotation=(180, 0, 0)  # Flipped
        )
        
        # Create render products
        self.render_products = []
        self.setup_render_products()
    
    def _create_ir_camera(self, name, prim_path, args):
        """Create an IR depth camera.
        
        Args:
            name: Camera name
            prim_path: Path for camera prim
            args: Configuration arguments
            
        Returns:
            Camera prim
        """
        # IR cameras: 91° FOV, depth range 0.2m - 2.5m
        camera = setup_camera(
            name,
            prim_path,
            width=args.depth_width,
            height=args.depth_height,
            fov=91.0
        )
        
        # Set clipping range for depth
        cam_geom = UsdGeom.Camera.Get(get_current_stage(), prim_path)
        cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.2, 2.5))
        
        return camera
    
    def _create_rgb_camera(self, name, prim_path, args):
        """Create an RGB camera.
        
        Args:
            name: Camera name
            prim_path: Path for camera prim
            args: Configuration arguments
            
        Returns:
            Camera prim
        """
        # RGB camera: 86° horizontal FOV (16:9 aspect)
        camera = setup_camera(
            name,
            prim_path,
            width=args.rgb_width,
            height=args.rgb_height,
            fov=86.0
        )
        
        return camera
    
    def setup_render_products(self):
        """Set up render products for all cameras."""
        # IR left (depth)
        self.ir_left_rp = setup_render_product(
            self.ir_left_camera_path,
            (self.args.depth_width, self.args.depth_height),
            f"{self.name}_ir_left_depth"
        )
        self.render_products.append({
            'name': f"{self.name}_ir_left_depth",
            'camera_path': self.ir_left_camera_path,
            'render_product': self.ir_left_rp,
            'type': 'depth'
        })
        
        # IR right (depth)
        self.ir_right_rp = setup_render_product(
            self.ir_right_camera_path,
            (self.args.depth_width, self.args.depth_height),
            f"{self.name}_ir_right_depth"
        )
        self.render_products.append({
            'name': f"{self.name}_ir_right_depth",
            'camera_path': self.ir_right_camera_path,
            'render_product': self.ir_right_rp,
            'type': 'depth'
        })
        
        # RGB camera (color + depth)
        self.rgb_camera_rp_rgb = setup_render_product(
            self.rgb_camera_path,
            (self.args.rgb_width, self.args.rgb_height),
            f"{self.name}_rgb_rgb"
        )
        self.render_products.append({
            'name': f"{self.name}_rgb_rgb",
            'camera_path': self.rgb_camera_path,
            'render_product': self.rgb_camera_rp_rgb,
            'type': 'rgb'
        })
        
        self.rgb_camera_rp_depth = setup_render_product(
            self.rgb_camera_path,
            (self.args.depth_width, self.args.depth_height),
            f"{self.name}_rgb_depth"
        )
        self.render_products.append({
            'name': f"{self.name}_rgb_depth",
            'camera_path': self.rgb_camera_path,
            'render_product': self.rgb_camera_rp_depth,
            'type': 'depth'
        })


class ZedMini(PrimObj):
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
        self.prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)
        super().__init__(self.prim_path, self.prim)

        # Left RGB camera
        self.left_camera_path = f"{self.prim_path}/{name}_left"
        self.left_camera = Camera(
            prim_path=self.left_camera_path,
            resolution=(self.rgb_width, self.rgb_height),
            frequency=frequency
        )
        self.left_camera.set_local_pose(
            [-self.baseline * 0.5, 0, 0],
            rot_utils_np.euler_angles_to_quats([90, 0, 0], degrees=True),
            camera_axes="usd"
        )


        # Right RGB camera
        self.right_camera_path = f"{self.prim_path}/{name}_right"
        self.right_camera = Camera(
            prim_path=self.right_camera_path,
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

class Gripper:
    """Dual camera gripper setup."""
    
    def __init__(self, name, parent_path, args):
        """Initialize the gripper with dual Orbbec Gemini 2 cameras.
        
        Args:
            name: Name of the gripper
            parent_path: Parent prim path
            args: OrbbecGemini2Args configuration
        """
        self.name = name
        self.args = args
        
        # Gripper geometry
        # Measured offset: 0.2764 * 0.5 = 0.1382m (half baseline between cameras)
        self.offset = 0.2764 * 0.5
        self.camera_rotation = 18  # 18° tilt from model
        
        # Create gripper empty transform
        self.prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)
        
        # Create left camera
        self.orbbec_left = OrbbecGemini2(f"{name}_left", self.prim_path, args)
        set_transform(
            self.orbbec_left.prim,
            translation=(self.offset, 0.0, 0.0),
            rotation=(0.0, self.camera_rotation, 180)
        )
        
        # Create right camera
        self.orbbec_right = OrbbecGemini2(f"{name}_right", self.prim_path, args)
        set_transform(
            self.orbbec_right.prim,
            translation=(-self.offset, 0.0, 0.0),
            rotation=(0.0, -self.camera_rotation, 180)
        )
        
        # Set gripper orientation
        set_transform(
            self.prim,
            translation=(0.0, 0.0, 0.0),
            rotation=(90.0, 0.0, 180.0)
        )
        
        self.current_parent = None
    
    def get_all_render_products(self):
        """Get all render products from both camera rigs.
        
        Returns:
            List of all render products
        """
        return self.orbbec_left.render_products + self.orbbec_right.render_products
