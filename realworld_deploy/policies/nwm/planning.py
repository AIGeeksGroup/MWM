# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import yaml
import os
import numpy as np
import lpips
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from diffusers.models import AutoencoderKL

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from isolated_nwm_infer import model_forward_wrapper
from misc import calculate_delta_yaw, get_action_torch, save_planning_pred, log_viz_single, transform, unnormalize_data
import distributed as dist
from models import CDiT_models
import platform
import pathlib
import sys
from torchvision.utils import save_image
from plot import show_images_with_labels
import time
from peft import PeftModel

if platform.system() == "Windows":
  temp = pathlib.PosixPath
  pathlib.PosixPath = pathlib.WindowsPath

with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])

class WM_Planning_Policy:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp = args.wm_eval.exp
        if platform.system() == "Windows":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        # Setting up Config
        # self.exp_eval = f'{self.exp}_nomad_eval' # local paths etc.
        self.exp_eval = self.exp
        self.get_eval_name()

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        self.config = default_config

        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        self.config.update(user_config)

        latent_size = self.config['image_size'] // 8
        self.latent_size = self.config['image_size'] // 8
        self.num_cond = self.config['eval_context_size']

        # logging directory
        #if self.args.save_preds:
        #    exp_name = os.path.basename(self.args.exp).split('.')[0]
        #    self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
        #    os.makedirs(self.args.save_output_dir, exist_ok=True)

        # Loading Model
        print("loading model...")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond,
            input_size=latent_size,
        )

        ckp = torch.load(f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.wm_eval.ckp}.pth.tar',
                         map_location='cpu', weights_only=False)
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = model
        if args.lora_dir != None:
            print("load lora.")
            self.model = PeftModel.from_pretrained(model, args.lora_dir).eval()
        #self.model = torch.compile(model)
        self.diffusion = create_diffusion(args.diffusion_infer_mode)
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        if platform.system() != "Windows" and args.lora_dir == None:
            self.model = torch.compile(model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device],
                                                               find_unused_parameters=False)
            self.model_without_ddp = self.model.module

        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mode = 'cem'  # assume CEM for planning
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 3  # hardcoded (delta_x, delta_y, delta_yaw)

    def init_mu_sigma(self, obs_0, traj_len):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim)
        mu[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale'])
        return mu, sigma
    
    def generate_actions(self, obs_image, goal_image, len_traj_pred=None):
        #idx_string = "_".join(map(str, idxs.flatten().int().tolist()))
        #image_plot_dir = os.path.join(dataset_save_output_dir, 'plots')
        #os.makedirs(image_plot_dir, exist_ok=True)
        if len_traj_pred is None:
            len_traj_pred = self.config["trajectory_eval_len_traj_pred"]

        #print("generate_actions ...", file=sys.__stdout__)

        n_evals = obs_image.shape[0]
        mu, sigma = self.init_mu_sigma(obs_image, len_traj_pred)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        #print("init_mu_sigma finished.", file=sys.__stdout__)

        for i in range(self.opt_steps):
            #print("opt_steps: %d" % i, file=sys.__stdout__)
            t1 = time.time()
            losses = []
            for traj in range(n_evals):
                #traj_id = int(idxs.flatten()[traj].item())
                sample = (torch.randn(self.num_samples, self.action_dim).to(self.device) * sigma[traj] + mu[traj])
                single_delta = sample[:, :2]
                deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)
                unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
                delta_yaw = calculate_delta_yaw(unnorm_deltas)
                deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
                deltas[:, -1, -1] += sample[:, -1] * (np.pi / 2)

                cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)
                cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.args.num_samples, 1, 1, 1, 1).squeeze(1)

                # WM is stochastic, so we can repeat the evaluation of each trajectory and average to reduce variance
                if self.num_repeat_eval * self.num_samples > 120:
                    cur_losses = []
                    for r in range(self.num_repeat_eval):
                        preds = self.autoregressive_rollout(cur_obs_image, deltas, self.args.rollout_stride)
                        preds = preds[:, -1]  # take the last predicted image
                        loss = self.loss_fn(preds.to(self.device), cur_goal_image.to(self.device)).flatten(0)
                        cur_losses.append(loss)

                    loss = torch.stack(cur_losses).mean(dim=0)
                else:
                    expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1)
                    expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)
                    expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1)

                    preds = self.autoregressive_rollout(expanded_obs_image, expanded_deltas, self.args.rollout_stride)
                    preds = preds[:, -1]

                    loss = self.loss_fn(preds.to(self.device), expanded_goal_image.to(self.device)).flatten(0)
                    loss = loss.view(self.num_repeat_eval, -1)
                    loss = loss.mean(dim=0)

                    preds = preds[:self.args.num_samples]

                sorted_idx = torch.argsort(loss)
                topk_idx = sorted_idx[:self.topk]
                topk_action = deltas[topk_idx][:, -1]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
            t2 = time.time()
            print("opt cost (secs):", t2 - t1)

                #if self.args.plot:
                #    self.visualize_trajectories(dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas,
                #                                cur_obs_image, cur_goal_image, preds, loss, topk_idx)

        # Final rollout
        deltas = mu[:, :2]
        deltas = deltas.unsqueeze(1).repeat(1, len_traj_pred, 1)

        # Calculate yaws
        unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)
        deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
        deltas[:, -1, -1] += mu[:, -1] * (np.pi / 2)

        best_preds = None
        min_loss = 100
        for i in range(0, 1):
            print("rollout final preds..")
            preds = self.autoregressive_rollout(obs_image, deltas, self.args.rollout_stride)
            loss = self.loss_fn(preds[:, -1].to(self.device), goal_image.squeeze(1).to(self.device)).flatten(0)
            if loss < min_loss:
                min_loss = loss
                best_preds = preds

        #if self.args.save_preds:
        #    save_planning_pred(dataset_save_output_dir, n_evals, idxs, obs_image, goal_image, preds, deltas, loss,
        #                       gt_actions)

        #if self.args.plot:
        #    img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
        #    plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_image.squeeze(1).to(self.device), idxs,
        #                     losses, save_path=img_name)

        pred_actions = get_action_torch(deltas[:, :, :2], ACTION_STATS_TORCH)
        # pred_yaw = deltas[:, :, -1].sum(1)

        pred_actions *= self.args.wm_eval.metric_waypoint_spacing
        pred_actions = torch.cat((pred_actions, deltas[:, :, -1:].to(pred_actions.device)), dim=-1)

        return pred_actions, loss.item(), best_preds

    def autoregressive_rollout(self, obs_image, deltas, rollout_stride):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)

        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i + 1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, self.args.rollout_stride,
                                                  self.latent_size, num_cond=self.num_cond, device=self.device)
            x_pred_pixels = x_pred_pixels.unsqueeze(1)

            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)  # append current prediction
            curr_obs = curr_obs[:, 1:]  # remove first observation
            preds.append(x_pred_pixels)

        preds = torch.cat(preds, 1)
        return preds

    def get_eval_name(self):
        # Get evaluation name for logging. Should overwrite for specific experiments
        self.eval_name = f'CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'

    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :2] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4))  # Define identity quaternion
        orientations_quat_wxyz[:, -1] = 1  # Define identity quaternion
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(positions_xyz=positions_xyz, orientations_quat_wxyz=orientations_quat_wxyz,
                                timestamps=timestamps)
        return traj

    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)

        result = main_ape.ape(traj_ref, traj_pred, est_name='traj',
                              pose_relation=PoseRelation.translation_part, align=False, correct_scale=False)
        ate = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
                              pose_relation=PoseRelation.rotation_angle_deg, align=False, correct_scale=False,
                              delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_rot = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
                              pose_relation=PoseRelation.translation_part, align=False, correct_scale=False,
                              delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_trans = result.stats['rmse']

        return ate, rpe_trans, rpe_rot

