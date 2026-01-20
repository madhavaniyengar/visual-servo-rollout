from functools import partial
from threading import Event, Thread
from typing import Dict, Any

import torch
import torch.nn as nn

from train import create_model, create_transforms
from datastructs import StereoSample
from modelutils import Transform, box2robot

def create_direction_model(config, experiment_config: Dict[str, Any]):
    model = create_model(experiment_config)
    transforms_dict = create_transforms(experiment_config)
    model.to("cuda")
    run_name = experiment_config['logging']['run_name']
    checkpoint_path = experiment_config["training"].get("checkpoint_dir", f"checkpoints/{run_name}")
    checkpoint = torch.load(f"{checkpoint_path}/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.Sequential(
        Transform(partial(StereoSample.transform, transforms=transforms_dict['eval'])), # not turning things correctly
        model,
        Transform(lambda modeloutput: modeloutput.pred),
        Transform(box2robot),
        Transform(lambda out : out.detach().cpu().numpy().squeeze())
    )
    return model, transforms_dict

class KeyboardFlags:

    def __init__(self):
        self._step_flag = Event()
        self._quit_flag = Event()
        Thread(target=self._dispatch, daemon=True).start()

    def _dispatch(self):
        while not self._quit_flag.is_set():
            if input().strip() == "q":
                self._quit_flag.set()
            else:
                self._step_flag.set()

    def clear_step_flag(self):
        self._step_flag.clear()

    def quit_flag(self):
        return self._quit_flag.is_set()

    def step_flag(self):
        return self._step_flag.is_set()
