class Robot:

    def __init__(self, name, parent_path, init_translation, init_rotation, offset_model):
        self.path = f"{parent_path}/{name}"
        self.empty = create_empty(self.path)
        self.next_direction_model = offset_model
        set_transform(self.path, init_translation, init_rotation)

        self.camera = ZedMini("camera", parent_path=self.path)
        self.body = VisualCuboid(
            prim_path=f"{self.path}/body",
            name="camera_body",
            size=0.05, # xform rel to parent is identity
            color=np.array([0, 255, 0]),
        )

def action_loop_once(robot):
    obs = get_obs(robot)
    dir = next_direction(robot, obs)
    move(dir, robot)
    return obs

def get_obs(robot) -> Observation:
    left = robot.camera.get_left_rgb()
    right = robot.camera.get_right_rgb()
    depth = robot.camera.get_depth()

    return Observation(
            Image(left, robot.camera.left_camera_path),
            Image(right, robot.camera.right_camera_path),
            depth,
    )

def get_image(camera) -> Image:
   return Image(camera.get_rgb(), camera.prim_path)

def get_pose(robot):
    return prim_get_pose(robot.path)

def prim_get_pose(prim):
    xform = UsdGeom.Xform(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return matrix.T

def next_direction(robot, observation: Observation) -> np.ndarray: 
    stereo_sample = observation.stereo_sample()
    direction = robot.next_direction_model(stereo_sample) # in robot coordinate frame
    robot2world = prim_get_pose(robot) # should be equivalent to prim_get_pose(robot.camera)
    assert np.allclose(prime_get_pose(robot.camera, robot2world))
    direction = transform_utils.rotate(direction, robot2world)
    return direction

def move(action, robot):
    robot_new_pose = transform_utils.add_translation(action.direction, get_pose(robot))
    set_transform(
            robot,
            transform_utils.get_translation(robot_new_pose),
            transform_utils.get_euler(robot_new_pose)
    )
    # camera.set_world_pose(position=camera_pos, orientation=CAMERA_QUAT, camera_axes="usd")
    # this hopefuly is identical
