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
from isaacsim import SimulationApp
from standalone_examples.benchmarks.validation.benchmark_sdg_validation import passed

simulation_app = SimulationApp({"headless": False})

import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.sensors.camera import Camera
from pxr import Sdf, UsdLux, UsdGeom, Gf

import datagen2_isaacsim.hardwares as hardwares
from model import GeometricServoing

# Add Ground Plane
GroundPlane(prim_path="/World/GroundPlane", z_position=0)

# Add Light Source
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)

camera = create_camera()

my_world = World(stage_units_in_meters=1.0)

my_world.reset()
camera.initialize()

def get_obs():
    return camera.get_color_rgba()

def get_target_point():
    pass

def image_to_cam(image_point):
    # image to cam transform
    pass

def cam_to_world(cam_point):
    # cam to world transform
    pass

# start the simulator
for i in range(100):
    # capture camera image
    observation = get_obs()

    # jeff's code here
    grasp_point_image = get_target_point()
    
    # get point in cam frame
    grasp_point_cam = image_to_cam(grasp_point_image)
    # this will be used to ensure camera pointed towards target point
    grasp_point_world = cam_to_world(grasp_point_cam)
    
    # get direction in cam frame
    direction = np.linalg.inv(K) @ grasp_point_cam
    direction = direction / np.linalg.norm(direction)
    
    # convert direction to world frame
    direction = cam_to_world(grasp_point_cam)
    
    # take a step in the direction
    camera_pos = camera_pos + direction
    camera_x, camera_y, camera_z = camera_pos
    
    camera_prim = stage.GetPrimAtPath("/World/camera")
    xform = UsdGeom.Xformable(camera_prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == 'xformOp:translate':
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(camera_x, camera_y, camera_z))
    
    # always point camera toward current target point in world
    look_vec = grasp_point_world - np.array([camera_x, camera_y, camera_z])
    look_vec = look_vec / (np.linalg.norm(look_vec) + 1e-8)
    yaw = np.arctan2(look_vec[1], look_vec[0]) * 180.0 / np.pi
    
    # Convert to quaternion
    quat = rot_utils.euler_angles_to_quats(np.array([0, yaw, 0]), degrees=True)
    
    # Update rotation
    rotate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == 'xformOp:orient':
            rotate_op = op
            break
    if rotate_op is None:
        rotate_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, UsdGeom.XformOp.OpTypeRotateQuat)
    rotate_op.Set(Gf.Quatd(quat[0], quat[1], quat[2], quat[3]))
    
    # Update visual marker position to match camera position
    camera_marker.set_world_pose(position=np.array([camera_x, camera_y, camera_z]))
    
    my_world.step(render=True)

# shutdown the simulator automatically
simulation_app.close()
