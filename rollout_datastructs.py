from dataclasses import dataclass
from typing import List

import numpy as np

from datastructs import StereoSample

@dataclass 
class Image:
    image: np.ndarray
    path: str

@dataclass
class Observation:
    left: Image
    right: Image
    depth: np.ndarray

    def stereo_sample(self):
        return StereoSample(
            self.left.image,
            self.right.image,
            self.depth,
            None,
            None,
            None,
            None
        )

@dataclass 
class Action:
    direction: np.ndarray


@dataclass 
class Step:
    images: List[Image]

@dataclass 
class ObsAction:
    obs: Observation
    action: Action
