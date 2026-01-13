from scipy.spatial.transform import Rotation as R
import numpy as np

def add_translation(vec, pose_matrix):
    pose_matrix = pose_matrix.copy()
    pose_matrix[:3, -1] += vec
    return pose_matrix

def rotate(vec, pose_matrix):
    rot = pose_matrix[:3, :3]
    return np.dot(rot, vec)

def get_translation(pose_matrix):
    return pose_matrix[:3, -1]

def get_euler(pose_matrix):
    return R.from_matrix(pose_matrix[:3, :3]).as_euler("xyz", degrees=True)
