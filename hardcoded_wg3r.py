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

model = TransformOutputs(
    GeometricServoing(
        annotation_ndc=torch.as_tensor([0., 1.]),
        corner_model=prepare_corner_model(),
        os.path.abspath("../calibs.json"),
        "cuda"
    ),
    lambda output : GeometricServoing.grasp2cam_transform(output)[..., :3, -1]
)

def get_direction(left_img, cam2world):
    direction_camera = model(left_img)
    assert len(direction_camera.shape) == 2, "Output of model isn't batched huh??"
    assert direction_camera.shape[0] == 1, "Supposed to only be one direction as output??"
    assert cam2world.shape[-1] == 4, "Not an actual cam2world??"
    direction_world = (
            torch.einsum(
                'ij,bj->bi',
                cam2world[:3, :3],
                direction_camera
            )
    )
    assert direction_world.shape[0] == 1, "Only one vector is expected??"
    return direction_world.squeeze()[:3]

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
from pxr import Sdf, UsdLux, UsdGeom, Gf
from isaacsim.core.utils.viewports import set_camera_view
# Create a fresh new stage first to avoid render product conflicts
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

# Add Ground Plane
GroundPlane(prim_path="/World/GroundPlane", z_position=0)

distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(1000)
domeLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
domeLight.CreateIntensityAttr(500)


camera_initial_position = np.array([-2.0, -0.2, 0.3])
camera = Camera(
    prim_path="/World/camera",
    position=camera_initial_position,
    frequency=20,
    resolution=(1920, 1080),
    orientation=rot_utils_np.euler_angles_to_quats(np.array([0, 0, 10]), degrees=True),
)

my_world = World(stage_units_in_meters=1.0)

camera_marker = my_world.scene.add(
    VisualCuboid(
        prim_path="/World/camera_marker",
        name="camera_marker",
        position=camera_initial_position,
        size=0.15,
        color=np.array([255, 0, 0]),
    )
)

my_world.reset()
my_world.step(render=False)
for _ in range(5):
    simulation_app.update()
camera.initialize()


def get_image():
    return camera.get_rgb()

def get_depth():
    return camera.get_depth()

# this will be jeff's API. For now, hardcode the target points
target_points = [
    np.array([0.1, -0.4, 0.1]),
    np.array([0.1, -0.3, 0.1]),
    np.array([0.1, -0.2, 0.1]),
    np.array([0.1, -0.1, 0.1]),
    np.array([0.1, 0.0, 0.1]),
    ]
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

# start the simulation loop
STEP_SIZE = 0.1
IMAGES = []


print("Step 0 - Initial image")
my_world.step(render=True)
for _ in range(5):
    simulation_app.update() 
image = get_image()
IMAGES.append(image)

for i in range(5):
    print(f"Step {i+1}")

    grasp_point_image = get_target_point(image, i)
    
    # get point world frame. NOTE: uncomment this when the target points are in image coords
    grasp_point_world = image_to_world(grasp_point_image)
    # grasp_point_world = grasp_point_image
    # get current camera pos
    camera_pos_world, camera_quat_world = camera.get_world_pose()
    
    direction = grasp_point_world - camera_pos_world
    direction = direction / np.linalg.norm(direction)
    
    # move the camera in the direction of the target point while keeping it pointed towards the target point

    camera_pos = camera_pos_world + direction * STEP_SIZE
    camera.set_world_pose(position=camera_pos, orientation=camera_quat_world)
    set_camera_view(camera_pos, grasp_point_world, '/World/camera')

    # Update visual marker position to match camera position
    camera_marker.set_world_pose(position=camera_pos, orientation=camera_quat_world)

    # Step the world to render the new camera position
    my_world.step(render=True)
    # Allow render to complete before capturing image
    for _ in range(5):
        simulation_app.update()
    
    # Capture camera image after rendering
    image = get_image()
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
