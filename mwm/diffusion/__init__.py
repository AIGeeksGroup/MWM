# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from .gaussian_diffusion import GaussianDiffusion
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps, SelfForcingDiffusion
from .diffusion_utils import decode_latent_video
import torch
import os


def create_diffusion(
    timestep_respacing,
    diffusion_type="SpacedDiffusion",
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    real_score=None,
    fake_score=None
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    if diffusion_type == "SelfForcingDiffusion":
        return SelfForcingDiffusion(
            #use_timesteps=[1000, 750, 500, 250, 0],
            use_timesteps=range(1000, 0, -100),
            #use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            real_score=real_score,
            fake_score=fake_score
            # rescale_timesteps=rescale_timesteps,
        )
    elif diffusion_type == "SpacedDiffusion":
        return SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type
            # rescale_timesteps=rescale_timesteps,
        )
    else:
        return GaussianDiffusion(
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type
        )


class DiffusionVisualizer():
    def __init__(self, diffusion, save_dir):
        self.diffusion = diffusion
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_diffusion(self, model, target, ctx, actions, rel_time, vae, frame_i=0):
        import matplotlib.pyplot as plt
        import mediapy
        device = target.device
        B, T = target.shape[:2]
        act = actions.flatten(0, 1)
        rt = rel_time.flatten(0, 1)
        ctx = ctx.unsqueeze(1).expand(B, T, *ctx.shape[1:5]).flatten(0, 1)
        model_kwargs = dict(y=act, x_cond=ctx, rel_t=rt)
        z = torch.randn(B * T, 4, 28, 28, device=device)

        x_ts = []
        x_0s = []
        for i, out in enumerate(self.diffusion.ddim_sample_loop_progressive(
            model,
            z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
            progress=True,
        )):
            t = self.diffusion.num_timesteps - i

            x_t = decode_latent_video(vae, out["sample"].unflatten(0, (B, T)))[0, frame_i]
            x_0 = decode_latent_video(vae, out["pred_xstart"].unflatten(0, (B, T)))[0, frame_i]

            x_ts.append(x_t.cpu().detach().numpy().astype('uint8'))
            x_0s.append(x_0.cpu().detach().numpy().astype('uint8'))

        mediapy.write_video(os.path.join(self.save_dir, f"xt.mp4"),
                            x_ts, fps=10)
        mediapy.write_video(os.path.join(self.save_dir, f"x0.mp4"),
                            x_0s, fps=10)

