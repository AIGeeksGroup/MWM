import hydra
import os
import warnings
from dataclasses import dataclass
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
from plot import show_images_with_labels, save_videos

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

@dataclass
class Best:
    loss: float = float("inf")
    action: object = None
    look_deg: object = None
    obs_seq: object = None
    preds: object = None

    def auto_update(self, loss, action, look_deg, obs_seq, preds):
        if loss < self.loss:
            self.loss = loss
            self.action = action
            self.look_deg = look_deg
            self.obs_seq = obs_seq
            self.preds = preds


def eval(args, env, agent, gradient_step):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_steps = []
    episode_success = []
    episode_ne = []
    save_video = True
    save_traj = True

    for i in range(args.eval_episodes):

        print(f"eval episodes {i}:")

        cam_id = 0
        success = 0
        look_deg = 0
        best = Best()
        obs, t, goal, init_goal_dist = env.reset() # {obs_name: (obs_steps, obs_dim)}
        goal_image = imgTransform(Image.fromarray(goal[cam_id], mode='RGB')).unsqueeze(0).unsqueeze(0)
        save_img(goal[cam_id], os.path.join(args.work_dir, f"goal_img/"), f'episode_{i}.png')
        #print("init_goal_dist:", init_goal_dist)
        if init_goal_dist > 10:
            print("too far!")
            continue

        while t < args.max_episode_steps and abs(look_deg) < 360:

            # generate action for current direction
            print(f"look to {look_deg}°")
            for k in obs.keys():
                if k.startswith("image%d" % cam_id):
                    obs_seq = obs[k].unsqueeze(0)  # (obs_steps, obs_dim)
                    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                        pred_actions, detect_loss, preds = agent.generate_actions(obs_seq, goal_image)
                        best.auto_update(detect_loss, pred_actions, look_deg, obs_seq, preds)
                        print("detect_loss:", detect_loss)
                        if save_traj:
                            show_images_with_labels(
                                [obs_seq.squeeze(0)[-1:], preds.squeeze(0), goal_image.squeeze(0)],
                                args.work_dir,
                                f'episode{i}_look_to_{look_deg}.png'
                            )
                else:
                    pass
            if detect_loss > 0.35 and abs(look_deg + args.detect_turn_degree) < 360:
                obs, _, _, _ = env.step(np.array([[0, 0, args.detect_turn_degree]]))
                look_deg += args.detect_turn_degree
                continue

            # turn to best
            env.step(np.array([[0, 0, best.look_deg - look_deg]]))

            # preprocess action
            action_pred = best.action.detach().to('cpu').numpy()  # (1,horizon, action_dim) dim=0在训练时是Batchsize，在推理时是env_num
            action_pred[..., 2] = np.degrees(action_pred[..., 2])
            start = 0
            end = start + args.action_steps
            action = np.squeeze(action_pred[:, start:end, :], axis=0) # 多一个env_num维度

            # execute action
            print("move ...")
            _, success, move_success, metric_data = env.step(action)
            t += args.action_steps

            if success:
                break

        if not move_success:
            print(f"Master, base motion error detected!")
            continue

        if save_traj:
            observe_imgs = torch.stack([imgTransform(Image.fromarray(videos[cam_id], mode='RGB')) for videos in env.step_record])
            show_images_with_labels(
                [best.obs_seq.squeeze(0)[-1:], best.preds.squeeze(0), observe_imgs, goal_image.squeeze(0)],
                args.work_dir,
                f'move_record_for_episode{i}.png'
            )

        #if success and save_video:
        if save_video:
            print("save video ...")
            save_fps = 1
            save_videos(best.preds.squeeze(0), save_fps, os.path.join(args.work_dir, "videos/predict/"), f"episode_{i}.mp4")
            save_videos(observe_imgs, save_fps, os.path.join(args.work_dir, "videos/excute/"), f"episode_{i}.mp4")

        ne = metric_data["ne"]
        episode_steps.append(t)
        episode_success.append(success) if success==1 else episode_success.append(0)
        episode_ne.append(ne)

        print(f'[Episode {i}] success:{success},  NE:{ne}')


    print(f"Mean step: {np.nanmean(episode_steps)} Mean success: {np.nanmean(episode_success)}  Mean NE: {np.nanmean(episode_ne)}")
    return {'mean_step': np.nanmean(episode_steps), 'mean_success': np.nanmean(episode_success), 'mean_ne': np.nanmean(episode_ne)}

from omegaconf import DictConfig
@hydra.main(config_path="./config", config_name="navigation")
def main(args: DictConfig):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    print("creat env ...", file=sys.__stdout__)
    env = Env(args, imgTransform)
    
    # --------------- Create WM Policy -----------------
    print("creat wm policy ...", file=sys.__stdout__)
    agent = WM_Planning_Policy(args)

    metrics = {'step': 0}
    metrics.update(eval(args, env, agent, 0))
    logger.log(metrics, category='eval')


if __name__ == "__main__":
    main()









    

