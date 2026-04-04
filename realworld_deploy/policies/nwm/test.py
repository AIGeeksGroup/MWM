import hydra
import os
import warnings
warnings.filterwarnings('ignore')

import pathlib
import numpy as np
import torch
from torchvision import transforms

from policies.dp.utils import set_seed, Logger, report_parameters

from policies.nwm.env import Env
from PIL import Image
from planning import WM_Planning_Policy
import torchvision.transforms.functional as TF
import sys
import math
from utils import save_img
from plot import show_images_with_labelsV2
from misc import get_action_torch
import yaml

with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])

IMAGE_ASPECT_RATIO = (4 / 3)  # all images are centered cropped to a 4:3 aspect ratio in training

class CenterCropAR:
    def __init__(self, ar: float = IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            img = TF.center_crop(img, (h, int(h * self.ar)))
        else:
            img = TF.center_crop(img, (int(w / self.ar), w))
        return img

imgTransform = transforms.Compose([
    CenterCropAR(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

from omegaconf import DictConfig

@hydra.main(config_path="./config", config_name="navigation")
def walk_in_sim(args: DictConfig):
    import importlib
    module = importlib.import_module(args.task_path.replace("/", ".").replace(".py", ""))
    SimNode = getattr(module, "SimNode")  # SimNode
    cfg = getattr(module, "cfg")  # cfg
    simnode = SimNode(cfg)
    while True:
        simnode.render()

@hydra.main(config_path="./config", config_name="navigation")
def test_imagine2execute_gap(args: DictConfig):
    set_seed(args.seed)
    metric_waypoint_spacing = 0.255

    pi = math.pi

    '''

    # 不动
    action = np.array([[-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0]])
    '''
    # 旋转
    action = np.array([[-1.0 / 3, 0, pi/8],
                       [-1.0 / 3, 0, pi/8],
                       [-1.0 / 3, 0, -pi/8],
                       [-1.0 / 3, 0, -pi/8],
                       [-1.0 / 3, 0, -pi/8],
                       [-1.0 / 3, 0, -pi/8],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0]])

    
    # 左移右移
    action = np.array([[-1.0 / 3, -5, 0],
                       [-1.0 / 3, -5, 0],
                       [-1.0 / 3, -5, 0],
                       [-1.0 / 3, -5, 0],
                       [-1.0 / 3, -5, 0],
                       [-1.0 / 3, -5, 0],
                       [-1.0 / 3, 0, 0],
                       [-1.0 / 3, 0, 0]])
    # 前进后退
    action = np.array([[1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])
    
    cam_id = 0

    # ---------------- Create Environment ----------------
    print("creat env ...", file=sys.__stdout__)
    env = Env(args, imgTransform)

    # --------------- Create WM Policy -----------------
    print("creat wm policy ...", file=sys.__stdout__)
    agent = WM_Planning_Policy(args)

    obs, t = env.reset()  # {obs_name: (obs_steps, obs_dim)}

    obs_seq = obs['image%d' % cam_id].unsqueeze(0)  # (obs_steps, obs_dim)
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        acts = torch.from_numpy(action).unsqueeze(0)
        preds = agent.autoregressive_rollout(obs_seq, acts, args.rollout_stride)

    # action单位对齐到仿真环境
    pred_actions = get_action_torch(acts[:, :, :2], ACTION_STATS_TORCH).squeeze(0)
    pred_actions *= metric_waypoint_spacing
    pred_yaw = acts[:, :, -1]
    degyaw_pred = torch.from_numpy(np.degrees(pred_yaw.detach().to('cpu').numpy())).squeeze(0).unsqueeze(1)

    actions_for_env = torch.cat([pred_actions, degyaw_pred], dim=1)
    print("actions_for_env:", actions_for_env.shape)
    print(actions_for_env)

    env.step(actions_for_env)
    observe_imgs = torch.stack([imgTransform(Image.fromarray(env.video_list[t][cam_id], mode='RGB')) for t in range(len(env.video_list))])
    #observe_imgs = torch.from_numpy(np.zeros((8, 3, 224, 224)))
    print(obs_seq[:,0].shape)
    print(preds.shape)
    print(observe_imgs.shape)
    show_images_with_labelsV2(obs_seq[:,0], preds, observe_imgs, "./", "imagine2execute.png")


if __name__ == "__main__":
    test_imagine2execute_gap()
    #walk_in_sim()