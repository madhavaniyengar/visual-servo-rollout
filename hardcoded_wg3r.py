# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from isaacsim import SimulationApp
import os

from model import GeometricServoing
from CornerDetector.cornerdetector import prepare_corner_model, TransformOutputs
import torchvision.transforms.v2 as v2
from dataset import StereoSample
import imageio


simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
})



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
# Create a fresh new stage first to avoid render product conflicts

# CAMERA_QUAT = rot_utils_np.euler_angles_to_quats(np.array([0, 0, 90]), degrees=True)
# Desired orientation: 90 degrees roll (X), 0 pitch (Y), 0 yaw (Z) in world coordinates
DESIRED_EULER = np.array([90, 0, 0])  # Roll, Pitch, Yaw in degrees
CAMERA_QUAT = rot_utils_np.euler_angles_to_quats(DESIRED_EULER, degrees=True)

EXTERNAL_EULER = np.array([70, 0, 0])
EXTERNAL_QUAT = rot_utils_np.euler_angles_to_quats(EXTERNAL_EULER, degrees=True)
EXTERNAL_CAMERA_POS = np.array([0.15, -0.75, 0.35])
# print(f"Desired camera Euler angles (roll, pitch, yaw): {DESIRED_EULER}")
# print(f"Desired camera quaternion (w, x, y, z): {CAMERA_QUAT}")

omni.usd.get_context().new_stage()
for _ in range(10):
    simulation_app.update()

# Get the fresh stage
stage = omni.usd.get_context().get_stage()

# Load the scene from USDZ file
scene_path = 'visual-servo-rollout/output_scene.usdz'
world_prim = stage.DefinePrim("/World", "Xform")
world_prim.GetReferences().AddReference(scene_path, "/World")
print(f"Loaded /World from USDZ as reference")
for _ in range(20):
    simulation_app.update()

# Add Ground Plane
# GroundPlane(prim_path="/World/GroundPlane", z_position=0)

distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(1000)
domeLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
domeLight.CreateIntensityAttr(500)


# camera_initial_position = np.array([-2.0, -0.2, 0.3])
# camera_initial_position = np.array([0.1, -0.5, 0.5])
camera_initial_position = np.array([0.15, -0.4, 0.15])


my_world = World(stage_units_in_meters=1.0)
my_world.reset()

# Add green cylinder prim under world
cylinder_euler = np.array([90, 0, 0])  # Roll, Pitch, Yaw in degrees
cylinder_quat = rot_utils_np.euler_angles_to_quats(cylinder_euler, degrees=True)
green_cylinder = VisualCylinder(
    prim_path="/World/green_cylinder",
    name="green_cylinder",
    translation=np.array([0.1, 0.055, 0.05]),
    orientation=cylinder_quat,
    scale=np.array([0.01, 0.01, 0.001]),
    color=np.array([0, 255, 0]),  # Green color
)

camera_marker = my_world.scene.add(
    VisualCuboid(
        prim_path="/World/camera_marker",
        name="camera_marker",
        position=camera_initial_position,
        size=0.05,
        color=np.array([0, 255, 0]),
    )
)
camera = Camera(
    prim_path="/World/camera",
    position=camera_initial_position,
    frequency=20,
    resolution=(1920, 1080),
    orientation=CAMERA_QUAT,
)
external_camera = Camera(
    prim_path="/World/external_camera",
    position=EXTERNAL_CAMERA_POS,
    frequency=20,
    resolution=(1920, 1080),
    orientation=EXTERNAL_QUAT,
)

camera.initialize()
external_camera.initialize()
camera.set_clipping_range(0.1, 100.0)
external_camera.set_clipping_range(0.1, 100.0)

my_world.step(render=False)
for _ in range(25):
    simulation_app.update()

camera.add_distance_to_image_plane_to_frame()

# Set focal length to fixed value
focal_length = 2.8  # mm
camera.set_focal_length(focal_length)
camera.set_horizontal_aperture(5.23)
camera.set_vertical_aperture(5.23 / (1920/1080))

# IMPORTANT: Set the camera pose with camera_axes="usd" to prevent auto-rotation
# This ensures the camera respects world coordinates (90, 0, 0) without Isaac Sim's default +Y up rotation
camera.set_world_pose(position=camera_initial_position, orientation=CAMERA_QUAT, camera_axes="usd")

external_camera.set_world_pose(position=EXTERNAL_CAMERA_POS, orientation=EXTERNAL_QUAT, camera_axes="usd")
def opencv_to_opengl_vector(vector_opencv):
    """
    Convert a vector from OpenCV camera coordinate frame to OpenGL camera coordinate frame.
    
    Args:
        vector_opencv: numpy array of shape (3,) or (N, 3) in OpenCV camera coordinates
        
    Returns:
        numpy array of same shape in OpenGL camera coordinates
    """
    # Transformation matrix: flip Y and Z axes
    T = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    if vector_opencv.ndim == 1:
        # Single vector
        return T @ vector_opencv
    else:
        # Batch of vectors
        return (T @ vector_opencv.T).T


model = GeometricServoing(
        annotation_ndc=torch.as_tensor([0., 1.]),
        corner_model=prepare_corner_model(device="cuda:0"),
        path_to_camera_calibs=os.path.abspath("calibs.json"),
        device="cuda:0"
    )

def get_direction(left_img, left_depth,cam2world):
    left_img = left_img[None, ...]
    left_depth = left_depth[None, ...]
    transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]

            )
        ])
    sample = StereoSample(left_img=left_img, right_img=left_img, left_depth=left_depth, right_depth=left_depth, offset=torch.zeros(1, 3))
    sample.transform(transform)
    sample.move_to("cuda:0")
    with torch.no_grad():
        direction_camera = model(sample)
    assert len(direction_camera.shape) == 2, "Output of model isn't batched huh??"
    assert direction_camera.shape[0] == 1, "Supposed to only be one direction as output??"
    assert cam2world.shape[-1] == 4, "Not an actual cam2world??"

    direction_world = (
            torch.einsum(
                'ij,bj->bi',
                torch.as_tensor(cam2world[:3, :3]).to("cuda:0").float(),
                direction_camera
            )
    )
    assert direction_world.shape[0] == 1, "Only one vector is expected??"
    direction_opengl = opencv_to_opengl_vector(direction_world.squeeze()[:3].cpu().numpy())
    return direction_opengl


def get_cam2world():
    prim = camera.prim
    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(matrix).T


def get_image(camera_obj):
    return camera_obj.get_rgb()

def get_depth(camera_obj):
    return camera_obj.get_depth()

target_points = [
    np.array([0.1, 0.1, 0.061 + 0.045]),
    ] * 50
def get_target_point(image, i):
    return target_points[i]

def image_to_world(image_point):
    """
    image_point (1, 2) point in image coords

    returns (1, 3) point in world coords
    """
    depth_map = get_depth()
    depth = depth_map[image_point[1], image_point[0]]
    return camera.get_world_points_from_image_coords(
        image_point,
        depth
    )

BOX_POS_X = 0.1
BOX_POS_Y = 0.1
BOX_POS_Z = 0.045

# start the simulation loop
STEP_SIZE = 0.01
NUM_STEPS = 35
IMAGES = []


print("Step 0 - Initial image")
my_world.step(render=True)
for _ in range(5):
    simulation_app.update() 
image = get_image(camera)
IMAGES.append(image)
EXTERNAL_IMAGES = []

def direction_to_quaternion(direction):
    """
    Compute quaternion that rotates Y-axis (0, 1, 0) to align with direction vector.
    Returns quaternion in (w, x, y, z) format.
    """
    direction = direction / np.linalg.norm(direction)
    # First, rotate 90 degrees about X-axis to get from Y-axis to Z-axis
    # 90 degrees = π/2 radians, quaternion for rotation about X-axis
    x_rot_90 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])  # (w, x, y, z)
    
    # Now compute rotation from Z-axis to direction vector
    z_axis = np.array([0, 0, 1])
    
    # Handle case where direction is parallel to Z-axis
    dot = np.dot(z_axis, direction)
    if abs(dot - 1.0) < 1e-6:
        # Already aligned with Z
        return x_rot_90
    elif abs(dot + 1.0) < 1e-6:
        # Opposite to Z
        z_to_dir_quat = np.array([0, 1, 0, 0])  # 180 degree rotation around X
    else:
        # Compute rotation axis and angle
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1, 1))
        
        # Convert axis-angle to quaternion
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis
        z_to_dir_quat = np.array([w, xyz[0], xyz[1], xyz[2]])
    
    # Compose quaternions: q_result = q2 * q1 (apply q1 first, then q2)
    # q = q2 * q1 = (w2*w1 - x2*x1 - y2*y1 - z2*z1,
    #                w2*x1 + x2*w1 + y2*z1 - z2*y1,
    #                w2*y1 - x2*z1 + y2*w1 + z2*x1,
    #                w2*z1 + x2*y1 - y2*x1 + z2*w1)
    q1 = x_rot_90
    q2 = z_to_dir_quat
    w = q2[0]*q1[0] - q2[1]*q1[1] - q2[2]*q1[2] - q2[3]*q1[3]
    x = q2[0]*q1[1] + q2[1]*q1[0] + q2[2]*q1[3] - q2[3]*q1[2]
    y = q2[0]*q1[2] - q2[1]*q1[3] + q2[2]*q1[0] + q2[3]*q1[1]
    z = q2[0]*q1[3] + q2[1]*q1[2] - q2[2]*q1[1] + q2[3]*q1[0]
    return np.array([w, x, y, z])


# try:
#     while simulation_app.is_running():
#         # Update the simulation app to process window events and keep rendering
#         simulation_app.update()
#         # Optionally step the world if you want physics to continue
#         # my_world.step(render=True)
# except KeyboardInterrupt:
#     print("\nShutting down simulation...")


for i in range(NUM_STEPS):
    print(f"Step {i+1}")

    image = get_image(camera)
    depth = get_depth(camera)
    cam2world = get_cam2world()


    # save the image to a file
    image_path = f"output/image_{i}.png"
    imageio.imwrite(image_path, image)
    print(f"Saved image to {image_path}")

    arrowhead = VisualCone(
    prim_path="/World/arrowhead",
    name="arrowhead",
    position=camera_initial_position,
    radius=0.01,  # Base radius of the cone
    height=0.02,  # Height of the cone
    color=np.array([255, 0, 0]),  # Red color to match arrow
    visible=True
    )


    direction = get_direction(image, depth, cam2world)
    direction = direction / np.linalg.norm(direction)


    
    # get current camera pos
    camera_pos_world, _ = camera.get_world_pose()

    draw_iface = _debug_draw.acquire_debug_draw_interface()
    arrow_length = 0.5  # Adjust this to change arrow length
    arrow_end = camera_pos_world + direction * arrow_length

    # Clear previous lines (optional - comment out if you want to see history)
    draw_iface.clear_lines()

    # Create carb.Float3 and carb.ColorRgba objects
    start_point = carb.Float3(camera_pos_world[0], camera_pos_world[1], camera_pos_world[2])
    end_point = carb.Float3(arrow_end[0], arrow_end[1], arrow_end[2])
    arrow_color = carb.ColorRgba(1.0, 0.0, 0.0, 1.0)  # Red arrow (r, g, b, a)
    arrow_thickness = 3.0  # Line thickness in pixels

    draw_iface.draw_lines(
        [start_point],  # List of carb.Float3
        [end_point],    # List of carb.Float3
        [arrow_color],  # List of carb.ColorRgba
        [arrow_thickness]  # List of float
    )

    arrowhead_quat = direction_to_quaternion(direction)
    arrowhead.set_world_pose(
        position=arrow_end,  # Position at the end of the arrow
        orientation=arrowhead_quat
    )

    camera_pos = camera_pos_world + direction * STEP_SIZE
    # Always pass orientation to maintain camera direction
    print(f"Camera position: {camera_pos_world}")
    print(f"Direction: {direction}")
    camera.set_world_pose(position=camera_pos, orientation=CAMERA_QUAT, camera_axes="usd")

    # Update visual marker position to match camera position
    camera_marker.set_world_pose(position=camera_pos)

    # Step the world to render the new camera position
    my_world.step(render=True)
    for _ in range(10):
        simulation_app.update()
    # Allow render to complete before capturing image
    image = get_image(camera)
    IMAGES.append(image)

    external_image = get_image(external_camera)
    EXTERNAL_IMAGES.append(external_image)

# save all images to a video
video_path = "output/camera_images.mp4"
imageio.mimsave(video_path, IMAGES[1:], fps=1)
print(f"Saved video to {video_path}")

video_path = "output/external_camera_images.mp4"
imageio.mimsave(video_path, EXTERNAL_IMAGES[1:], fps=1)
print(f"Saved video to {video_path}")

# Keep the simulation running for visualization
# Use update() to process window events and keep rendering


# shutdown the simulator
simulation_app.close()
