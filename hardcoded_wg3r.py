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
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.sensors.camera import Camera
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from isaacsim.core.utils.viewports import set_camera_view
# Create a fresh new stage first to avoid render product conflicts

# CAMERA_QUAT = rot_utils_np.euler_angles_to_quats(np.array([0, 0, 90]), degrees=True)
CAMERA_QUAT = rot_utils_np.euler_angles_to_quats(np.array([90, 0, 0]), degrees=True)

omni.usd.get_context().new_stage()
for _ in range(10):
    simulation_app.update()

# Get the fresh stage
stage = omni.usd.get_context().get_stage()

# Load the scene from USDZ file
scene_path = '/home/madhavai/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/visual-servoing-rollout/output_scene.usdz'
world_prim = stage.DefinePrim("/World", "Xform")
world_prim.GetReferences().AddReference(scene_path, "/World")
print(f"Loaded /World from USDZ as reference")
for _ in range(20):
    simulation_app.update()

# Search through the prim tree to find Box_0_0_0 and get its world pose
def find_prim_by_name(stage, target_name):
    """Traverse the prim tree to find a prim by name and print hierarchy"""
    found_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == target_name:
            found_prim = prim
            print(f"Found prim '{target_name}' at path: {prim.GetPath()}")
            
            # Print the hierarchy path
            parent = prim.GetParent()
            hierarchy = [prim.GetName()]
            while parent and parent.GetPath() != Sdf.Path("/"):
                hierarchy.insert(0, parent.GetName())
                parent = parent.GetParent()
            print(f"Hierarchy: /{'/'.join(hierarchy)}")
            break
    return found_prim

def get_prim_world_pose(prim):
    """Get the world pose (position and orientation) of a prim"""
    if not prim:
        return None, None
    
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        print(f"Prim {prim.GetPath()} is not xformable")
        return None, None
    
    # Get world transform matrix
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # Extract translation (position)
    translation = world_transform.ExtractTranslation()
    position = np.array([translation[0], translation[1], translation[2]])
    
    # Extract rotation as quaternion
    rotation = world_transform.ExtractRotation()
    quat = rotation.GetQuat()
    # Gf.Quatd is (w, x, y, z) format
    real = quat.GetReal()
    imag = quat.GetImaginary()
    orientation = np.array([real, imag[0], imag[1], imag[2]])  # (w, x, y, z)
    
    return position, orientation

# Find and print Box_0_0_0 pose
box_prim = find_prim_by_name(stage, "Box_0_0_0")
if box_prim:
    box_position, box_orientation = get_prim_world_pose(box_prim)
    print(f"\n=== Box_0_0_0 World Pose ===")
    print(f"Position (x, y, z): {box_position}")
    print(f"Orientation (w, x, y, z): {box_orientation}")
    
    # Also print Euler angles for easier interpretation
    if box_orientation is not None:
        euler = rot_utils_np.quats_to_euler_angles(box_orientation, degrees=True)
        print(f"Orientation (Euler degrees - roll, pitch, yaw): {euler}")
    print(f"=============================\n")
else:
    print("WARNING: Could not find prim 'Box_0_0_0' in the scene!")
    print("Available prims in scene:")
    for prim in stage.Traverse():
        print(f"  - {prim.GetPath()} ({prim.GetTypeName()})")

breakpoint()

# Add Ground Plane
# GroundPlane(prim_path="/World/GroundPlane", z_position=0)

distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(1000)
domeLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
domeLight.CreateIntensityAttr(500)


# camera_initial_position = np.array([-2.0, -0.2, 0.3])
# camera_initial_position = np.array([0.1, -0.5, 0.5])
camera_initial_position = np.array([0.05, -0.4, -0.15])
camera = Camera(
    prim_path="/World/camera",
    position=camera_initial_position,
    frequency=20,
    resolution=(1920, 1080),
    orientation=CAMERA_QUAT,
)

my_world = World(stage_units_in_meters=1.0)

# camera_marker = my_world.scene.add(
#     VisualCuboid(
#         prim_path="/World/camera_marker",
#         name="camera_marker",
#         position=camera_initial_position,
#         size=0.15,
#         color=np.array([255, 0, 0]),
#     )
# )

my_world.reset()
my_world.step(render=False)
for _ in range(5):
    simulation_app.update()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()

# Set focal length to fixed value
focal_length = 2.8  # mm
camera.set_focal_length(focal_length)
camera.set_horizontal_aperture(5.23)
camera.set_vertical_aperture(5.23 / (1920/1080))

# Get the initial camera orientation AFTER initialization - this is the correct orientation in "world" axes
_, CAMERA_ORIENTATION = camera.get_world_pose()
print(f"Initial camera orientation (world axes): {CAMERA_ORIENTATION}")

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
    return direction_world.squeeze()[:3].cpu().numpy()


def get_cam2world():
    prim = camera.prim
    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(matrix).T


def get_image():
    return camera.get_rgb()

def get_depth():
    return camera.get_depth()

# this will be jeff's API. For now, hardcode the target points
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

# def initialize_camera_pose(ee_range_x=[-0.05, 0.05], ee_range_y=[-0.4, -0.2], ee_range_z=[0.2, 0.25]):
#     np.random.seed(0)
#     camera_pos_x = np.random.uniform(ee_range_x[0], ee_range_x[1])
#     camera_pos_y = np.random.uniform(ee_range_y[0], ee_range_y[1])
#     camera_pos_z = np.random.uniform(ee_range_z[0], ee_range_z[1])
#     camera_pos = np.array([BOX_POS_X + camera_pos_x, BOX_POS_Y + camera_pos_y, BOX_POS_Z + camera_pos_z])
#     # Always pass orientation to maintain camera direction
#     camera.set_world_pose(position=camera_pos, orientation=CAMERA_ORIENTATION)

# start the simulation loop
STEP_SIZE = 0.05
NUM_STEPS = 5
IMAGES = []


print("Step 0 - Initial image")
my_world.step(render=True)
for _ in range(5):
    simulation_app.update() 
image = get_image()
IMAGES.append(image)

# initialize_camera_pose_hardcoded()


for i in range(NUM_STEPS):
    print(f"Step {i+1}")

    image = get_image()
    depth = get_depth()
    cam2world = get_cam2world()

    # save the image to a file
    image_path = f"output/image_{i}.png"
    imageio.imwrite(image_path, image)
    print(f"Saved image to {image_path}")


    direction = get_direction(image, depth, cam2world)
    direction = direction / np.linalg.norm(direction)


    
    # get current camera pos
    camera_pos_world, _ = camera.get_world_pose()

    camera_pos = camera_pos_world + direction * STEP_SIZE
    # Always pass orientation to maintain camera direction
    print(f"Camera position: {camera_pos_world}")
    print(f"Direction: {direction}")
    camera.set_world_pose(position=camera_pos, orientation=CAMERA_ORIENTATION)

    # Update visual marker position to match camera position
    # camera_marker.set_world_pose(position=camera_pos)

    # Step the world to render the new camera position
    image = my_world.step(render=True)
    for _ in range(10):
        simulation_app.update()
    # Allow render to complete before capturing image
    IMAGES.append(image)

# save images to an np array
valid_images = []
for image in IMAGES:
    if image is None:
        continue
    print(f"Image shape: {image.shape}")
    valid_images.append(image)
images_np = np.array(valid_images)
print("Images shape: ", images_np.shape)
os.makedirs('output', exist_ok=True)
np.save('output/camera_images.npy', images_np)
print(f"Saved images to output/camera_images.npy")

# Keep the simulation running for visualization
# Use update() to process window events and keep rendering
try:
    while simulation_app.is_running():
        # Update the simulation app to process window events and keep rendering
        simulation_app.update()
        # Optionally step the world if you want physics to continue
        # my_world.step(render=True)
except KeyboardInterrupt:
    print("\nShutting down simulation...")

# shutdown the simulator
simulation_app.close()
