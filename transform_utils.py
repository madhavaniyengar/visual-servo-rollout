from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

def add_translation(vec, pose_matrix):
    pose_matrix = pose_matrix.copy()
    pose_matrix[:3, -1] += vec
    return pose_matrix

def transform(vec, pose_matrix):
    return rotate(vec, pose_matrix) + get_translation(pose_matrix)

def rotate(vec, pose_matrix):
    rot = pose_matrix[:3, :3]
    return np.dot(rot, vec)

def get_translation(pose_matrix):
    return pose_matrix[:3, -1]

def get_euler(pose_matrix):
    return R.from_matrix(pose_matrix[:3, :3]).as_euler("xyz", degrees=True)

def resize_norm(vector: np.ndarray, norm: float):
    return vector / np.linalg.norm(vector) * norm

def create_se3(*, translation, parent2self_euler):
    rotn = np.transpose(R.from_euler("xyz", parent2self_euler, degrees=True).as_matrix())
    se3 = np.eye(4)
    se3[:3, :3] = rotn
    se3[:3, -1] = translation
    return se3
