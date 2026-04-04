# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import numpy as np
import torch as th
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import lpips
from .diffusion_utils import decode_latent_video, visualize_latent

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)

class SelfForcingDiffusion(SpacedDiffusion):
    def __init__(self, *args, rollout_len: int=8, context_len: int=4, fake_score=None, real_score=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_len = rollout_len
        self.context_len = context_len
        self.lpips_loss_fn = lpips.LPIPS(net='alex')

        self.fake_score = fake_score
        self.real_score = real_score

        self.fake_guidance_scale = 0
        self.real_guidance_scale = 0.0

        self.uncond_act = th.tensor([-1/3, -1/3, 0.0])

    def p_sample_skip(
            self,
            model,
            x,
            j,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t(j-1)} from the model at the given x_{tj}.
        :param model: the model to sample from.
        :param x: the current tensor at x_{tj}.
        :param j: the index of {t0, t1, ... tT}.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: - 'sample': a random sample from the model.

        """
        out = self.p_mean_variance(
            model,
            x,
            j,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        x_start = out["pred_xstart"]
        sample = self.q_sample(x_start, j-1, noise=noise)
        return {"sample": sample, "pred_xstart": x_start}

    def _sample_x0_at_step_s(self, model, shape, s: int, model_kwargs: dict, device, clip_denoised: bool = False, progress: bool =False, ddim=True):
        B = shape[0]
        img = th.randn(*shape, device=device)

        indices = range(len(self.timestep_map) - 1, s-1, -1)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for j in indices:
            if j != s:
                j = th.full((B,), j, device=device, dtype=th.long)
                with th.no_grad():
                    if not ddim:
                        sample = self.p_sample_skip(model, img, j, clip_denoised=clip_denoised, model_kwargs=model_kwargs)["sample"]
                    else:
                        sample = self.ddim_sample(model, img, j, clip_denoised=clip_denoised, model_kwargs=model_kwargs)["sample"]
                    img = sample
            else:
                j = th.full((B,), j, device=device, dtype=th.long)
                if not ddim:
                    out = self.p_mean_variance(model, img, j, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
                else:
                    out = self.ddim_sample(model, img, j, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        return out["pred_xstart"]

    def _compute_kl_grad(
        self, noisy_image: th.Tensor,
        estimated_clean_image: th.Tensor,
        timestep: th.Tensor,
        conditional_model_kwargs: dict,
        unconditional_model_kwargs: dict,
        normalization: bool = True,
        clip_denoised: bool = True,
        debug_dict: dict = None
    ) -> tuple[th.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image: a tensor with shape [B, C, H, W].
            - estimated_clean_image: a tensor with shape [B, C, H, W] representing the estimated clean image.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        pred_fake_image_cond = self.p_mean_variance(
            self.fake_score,
            noisy_image,
            timestep,
            clip_denoised=clip_denoised,
            model_kwargs=conditional_model_kwargs
        )["pred_xstart"]

        if self.fake_guidance_scale != 0.0:
            pred_fake_image_uncond = self.p_mean_variance(
                self.fake_score,
                noisy_image,
                timestep,
                clip_denoised=clip_denoised,
                model_kwargs=unconditional_model_kwargs
            )["pred_xstart"]
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)

        pred_real_image_cond = self.p_mean_variance(
            self.real_score,
            noisy_image,
            timestep,
            clip_denoised=clip_denoised,
            model_kwargs=conditional_model_kwargs
        )["pred_xstart"]

        if self.real_guidance_scale != 0.0:
            pred_real_image_uncond = self.p_mean_variance(
                self.real_score,
                noisy_image,
                timestep,
                clip_denoised=clip_denoised,
                model_kwargs=unconditional_model_kwargs
            )["pred_xstart"]
            pred_real_image = pred_real_image_cond + (
                pred_real_image_cond - pred_real_image_uncond
            ) * self.real_guidance_scale
        else:
            pred_real_image = pred_real_image_cond

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image - pred_real_image)
            normalizer = th.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = grad / normalizer
        grad = th.nan_to_num(grad)

        # debug
        # visualize_latent([[noisy_image, pred_real_image, pred_fake_image, estimated_clean_image]], debug_dict, "dmd")

        return grad, {
            "dmdtrain_gradient_norm": th.mean(th.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
            self,
            image: th.Tensor,
            conditional_model_kwargs: dict,
            unconditional_model_kwargs: dict,
            device,
            gradient_mask = None,
            debug_dict=None
    ) -> tuple[th.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image: a tensor with shape [B, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image
        B = image.shape[0]

        with th.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            j = int(th.randint(low=0, high=len(self.timestep_map), size=(1,), device=device).item())
            t = th.full((B,), j, device=device, dtype=th.long)

            noise = th.randn_like(image)
            noisy_latent = self.q_sample(image, t, noise=noise).detach()

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image=noisy_latent,
                estimated_clean_image=original_latent,
                timestep=t,
                conditional_model_kwargs=conditional_model_kwargs,
                unconditional_model_kwargs=unconditional_model_kwargs,
                debug_dict=debug_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            #dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            #), (original_latent.double() - grad.double()).detach(), reduction="mean")
            loss_map = 0.5 * F.mse_loss(original_latent.double(
             ), (original_latent.double() - grad.double()).detach(), reduction="none")
            dmd_loss = loss_map.mean(dim=tuple(range(1, loss_map.dim())))  # [B]
        return dmd_loss, dmd_log_dict

    def compute_framewise_dmd_loss(self, pred_seq, target, ctx, actions, rel_time, device, debug_dict):
        # ctx: [B, context_len, 4, 28, 28]
        # actions: [B, N, 3]
        # rel_time: [B, N]
        #print("dmd:", ctx.shape, actions.shape, rel_time.shape)
        B, T = pred_seq.shape[:2]
        frame_losses = []
        for i in range(T):
            if i != 0:
                debug_dict = None
                break
            image = pred_seq[:, i]
            act = actions[:, i]
            rt = rel_time[:, i] if rel_time is not None else None
            conditional_model_kwargs = dict(y=act, x_cond=ctx, rel_t=rt)
            unconditional_model_kwargs = dict(y=self.uncond_act.repeat(B, 1).to(device), x_cond=ctx, rel_t=rt)
            frame_loss = self.compute_distribution_matching_loss(
                image,
                conditional_model_kwargs,
                unconditional_model_kwargs,
                device,
                debug_dict=debug_dict
            )
            frame_losses.append(frame_loss[0])
            ctx = th.cat([ctx[:, 1:], target[:, i].unsqueeze(1)], dim=1)
        avg_loss = th.stack(frame_losses, dim=1).mean(dim=1)
        return avg_loss

    def training_losses(self, model, target, ctx, actions, rel_time, noise=None,
                        debug_dict=None, use_dmd_loss=False, save_training_rollout=False,
                        reweight_lpips=False):
        # Self-Forcing batch: (context, target, actions_delta, rel_time) 或 dict
        device = target.device
        B, N = target.shape[0:2]
        original_ctx = ctx

        s = int(th.randint(low=0, high=len(self.timestep_map), size=(1,), device=device).item())
        #s = 0
        preds = []
        for i in range(N):
            act = actions[:, i]
            rt = rel_time[:, i] if rel_time is not None else None
            model_kwargs = dict(y=act, x_cond=ctx, rel_t=rt)
            frame_shape = (B, *target[:, i].shape[1:])
            pred_x0 = self._sample_x0_at_step_s(model, frame_shape, s=s, model_kwargs=model_kwargs, device=device)
            preds.append(pred_x0)

            # rolling context + stop-gradient
            pred_detached = pred_x0.detach()
            #print("pred_detached shape:", pred_detached.shape)
            ctx = th.cat([ctx[:, 1:], pred_detached.unsqueeze(1)], dim=1)
            #if self.context_len is not None and ctx.shape[1] > self.context_len:
            #    ctx = ctx[:, -self.context_len :]

        vae = debug_dict["vae"]
        pred_seq = th.stack(preds, dim=1)  # [B, N, ...]
        mse = (pred_seq - target).pow(2).mean(dim=list(range(1, pred_seq.dim())))  # [B]

        if use_dmd_loss:
            dmd_loss = self.compute_framewise_dmd_loss(pred_seq, target, original_ctx, actions, rel_time, device, debug_dict)

        pred_seq_decode = decode_latent_video(vae, pred_seq.flatten(0, 1)).unflatten(0, (B, N)).permute(0, 1, 4, 2, 3)
        target_decode = decode_latent_video(vae, target.flatten(0, 1)).unflatten(0, (B, N)).permute(0, 1, 4, 2, 3)

        lpips_loss_fn = self.lpips_loss_fn.to(device)
        lpips_per = lpips_loss_fn(pred_seq_decode.flatten(0, 1), \
                                        target_decode.flatten(0, 1)) \
                                        .unflatten(0, (B, N))
        if reweight_lpips:
            lpips_per = lpips_per.flatten(2).mean(-1)
            w = th.arange(1, N + 1, device=lpips_per.device, dtype=lpips_per.dtype)
            w = w / w.sum()
            lpips_loss = (lpips_per * w).sum(dim=1)  # (B,)
        else:
            lpips_loss = lpips_per.mean(dim=1).squeeze(dim=(1, 2, 3))

        #pred = th.clip(vae.decode(pred_seq / 0.18215).sample.squeeze(0), -1., 1.)
        #pred = pred * 0.5 + 0.5

        terms = {"loss": lpips_loss, "mse": mse, "s": th.tensor([s], device=device, dtype=th.long)}

        # debug
        train_steps = debug_dict["train_steps"]
        save_dir = f'{debug_dict["save_dir"]}/rollout'

        if save_training_rollout and train_steps % 10 == 0:
            _, ax = plt.subplots(2, N, dpi=256)
            os.makedirs(save_dir, exist_ok=True)
            for i in range(0, N):
                pred = th.clip(vae.decode(pred_seq[0, i].unsqueeze(0) / 0.18215).sample.squeeze(0), -1., 1.)
                tar = th.clip(vae.decode(target[0, i].unsqueeze(0) / 0.18215).sample.squeeze(0), -1., 1.)
                pred = pred * 0.5 + 0.5
                tar = tar * 0.5 + 0.5
                ax[0, i].imshow((pred.permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8'))
                ax[1, i].imshow((tar.permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8'))
            plt.savefig(f'{save_dir}/{train_steps}.png')
            plt.close()

        return terms