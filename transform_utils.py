import numpy as np

def add_translation(vec, pose_matrix):
    pose_matrix = pose_matrix.copy()
    pose_matrix[:3, -1] += vec
    return pose_matrix

def rotate(vec, pose_matrix):
    rot = pose_matrix[:3, :3]
    return np.dot(rot, vec)
