# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

from isolated_nwm_infer import model_forward_wrapper
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib

matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt
import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL

from distributed import init_distributed
from models import CDiT_models
from diffusion import create_diffusion, DiffusionVisualizer
from datasets import SelfForcingTrainingDataset
from misc import transform
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def list_linear_module_names(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    _, rank, device, _ = init_distributed()
    # rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    latent_size = config['image_size'] // 8

    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    real_score_model = deepcopy(model).to(device)
    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema, False)
    requires_grad(real_score_model, False)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # load existing checkpoint
    latest_path = os.path.join(checkpoint_dir, f"{config['ckp']}.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
        if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
            raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
        latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)

        if "ema" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['ema'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            # model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            # res = ema.load_state_dict(model_ckp, strict=True)
            # print("Loading EMA model weights", res)
        else:
            pass
            # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

        real_model = None
        lora_finetune = args.add_lora_config or (args.start_from_lora != None)

        if args.dmd and "ema" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['ema'].items()}
            res = real_score_model.load_state_dict(model_ckp, strict=True)
            print("Loading real_score_model weights", res)
            real_model = PeftModel.from_pretrained(real_score_model, f"{checkpoint_dir}/teacher_lora").eval()

        if args.start_from_lora != None:
            print("load from lora.")
            model = PeftModel.from_pretrained(model, f"{checkpoint_dir}/{args.start_from_lora}", is_trainable=args.start_lora_trainable)
            model.print_trainable_parameters()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

        if args.add_lora_config:
            if args.start_from_lora != None:
                print("WARNING: Add another lora config from a pretrained lora model, are you sure?")

            # list layer 
            linear_names = list_linear_module_names(model)
            #print("\n".join(linear_names[:200]))
            #print("Total Linear:", len(linear_names))
            
            #target_modules = "all-linear"
            #target_modules = ["blocks.%d.attn.qkv" % i for i in range(25, 28)]
            target_modules = ["adaLN_modulation.1"]

            lora_config = LoraConfig(
                target_modules=target_modules,
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

        # if "opt" in latest_checkpoint:
        #     opt_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['opt'].items()}
        #     opt.load_state_dict(opt_ckp)
        #     print("Loading optimizer params")

        # if "epoch" in latest_checkpoint:
        #    start_epoch = latest_checkpoint['epoch'] + 1

        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]

        if "scaler" in latest_checkpoint:
            scaler.load_state_dict(latest_checkpoint["scaler"])

    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing=[10,15,20],
                                 diffusion_type="SelfForcingDiffusion",
                                 real_score=real_model,
                                 fake_score=model)  # default: 1000 steps, linear noise schedule
    eval_diffusion = create_diffusion(timestep_respacing=str(5))
    
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                goals_per_obs = int(data_config["goals_per_obs"])
                if data_split_type == 'test':
                    goals_per_obs = 4  # standardize testing

                if "distance" in data_config:
                    min_dist_cat = data_config["distance"]["min_dist_cat"]
                    max_dist_cat = data_config["distance"]["max_dist_cat"]
                else:
                    min_dist_cat = config["distance"]["min_dist_cat"]
                    max_dist_cat = config["distance"]["max_dist_cat"]

                if "len_traj_pred" in data_config:
                    len_traj_pred = data_config["len_traj_pred"]
                else:
                    len_traj_pred = config["len_traj_pred"]

                dataset = SelfForcingTrainingDataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    min_dist_cat=min_dist_cat,
                    max_dist_cat=max_dist_cat,
                    len_traj_pred=len_traj_pred,
                    context_size=config["context_size"],
                    normalize=config["normalize"],
                    transform=transform,
                    predefined_index=None,
                    traj_stride=1,
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    test_dataset.append(dataset)
                print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for gt_contexts, future_imgs, act, rel_time in loader:
            gt_contexts = gt_contexts.to(device, non_blocking=True)
            future_imgs = future_imgs.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)
            rel_t = rel_time.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    B, T1 = gt_contexts.shape[:2]
                    gt_contexts = gt_contexts.flatten(0, 1)
                    gt_contexts = tokenizer.encode(gt_contexts).latent_dist.sample().mul_(0.18215)
                    gt_contexts = gt_contexts.unflatten(0, (B, T1))

                    B, T2 = future_imgs.shape[:2]
                    future_imgs = future_imgs.flatten(0, 1)
                    future_imgs = tokenizer.encode(future_imgs).latent_dist.sample().mul_(0.18215)
                    future_imgs = future_imgs.unflatten(0, (B, T2))

                #dv = DiffusionVisualizer(diffusion, "./deffusion_vis")
                #dv.plot_diffusion(model, future_imgs, gt_contexts, act, rel_t, tokenizer)

                loss_dict = diffusion.training_losses(model, future_imgs, gt_contexts, act, rel_t,
                                                       debug_dict={"train_steps": train_steps,
                                                                   "vae": tokenizer, "save_dir": f"{experiment_dir}/self_forcing"},
                                                        save_training_rollout=False, reweight_lpips=config['reweight_lpips'])
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            if not bfloat_enable:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()

            # update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size() * gt_contexts.shape[0] * steps_per_sec
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0 and not lora_finetune:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                    if train_steps % (10 * args.ckpt_every) == 0 and train_steps > 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                        torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                elif rank == 0 and lora_finetune:
                    checkpoint_path = f"{checkpoint_dir}/lora_{train_steps}"
                    model.module.save_pretrained(checkpoint_path)
                    logger.info(f"Saved lora to {checkpoint_path}")

            if train_steps % args.eval_every == 0 and train_steps >= 0:
                ddim_eval, eval_diffusion = (True, eval_diffusion)
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                model.eval()
                # sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond)
                sim_score_0 = evaluate(model, tokenizer, eval_diffusion, test_dataset, rank, config["eval_batch_size"],
                                     config["num_workers"], latent_size, device, save_dir, args.global_seed,
                                     bfloat_enable, num_cond, eval_frame_index=0, ddim=ddim_eval)
                sim_score_7 = evaluate(model, tokenizer, eval_diffusion, test_dataset, rank, config["eval_batch_size"],
                                     config["num_workers"], latent_size, device, save_dir, args.global_seed,
                                     bfloat_enable, num_cond, eval_frame_index=7, ddim=ddim_eval)
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) frame:0  Perceptual Loss: {sim_score_0:.4f}, frame:7  Perceptual Loss: {sim_score_7:.4f}, Eval Time: {eval_time:.2f}")
                model.train()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


@torch.no_grad
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir,
             seed, bfloat_enable, num_cond, eval_batch=1, eval_frame_index=-1, ddim=False):

    def autoregressive_rollout(model, vae, diffusion, obs_image, deltas, rel_t, latent_size, device, ddim=False):
        # deltas: [B, 8, 3]
        # obs_image: [B, num_cond, 3, 224, 224]
        preds = []
        curr_obs = obs_image.clone().to(device)

        for i in tqdm(range(deltas.shape[1])):
            curr_delta = deltas[:, i:i + 1]
            curr_ret = rel_t[:, i]
            all_models = model, diffusion, vae
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps=8,
                                                  latent_size=latent_size, num_cond=num_cond, device=device, rel_t=curr_ret, ddim=ddim)
            x_pred_pixels = x_pred_pixels.unsqueeze(1)

            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)  # append current prediction
            curr_obs = curr_obs[:, 1:]  # remove first observation
            preds.append(x_pred_pixels)

        preds = torch.cat(preds, 1)
        return preds

    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    eval_cnt = 0
    for gt_contexts, future_imgs, act, rel_time in loader:
        gt_contexts = gt_contexts.to(device)
        future_imgs = future_imgs.to(device)
        rel_t = rel_time.to(device, non_blocking=True)
        act = act.to(device)
        T = future_imgs.shape[1]

        #print("evaluate:", act.shape, gt_contexts.shape, future_imgs.shape)
        # torch.Size([16, 3]) torch.Size([2, 4, 3, 224, 224]) torch.Size([2, 8, 3, 224, 224])

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            preds = autoregressive_rollout(model, vae, diffusion, gt_contexts, act, rel_t, latent_size, device, ddim=ddim)

            x_start_pixels = future_imgs[:, eval_frame_index]
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            preds = preds * 0.5 + 0.5
            pred = preds[:, eval_frame_index]

            res = eval_model(x_start_pixels, pred)
            score += res.sum()
            n_samples += len(res)
        eval_cnt += 1
        if eval_cnt == eval_batch:
            break

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(pred.shape[0], 10)):
            _, ax = plt.subplots(1, 2, dpi=256)
            ax[0].imshow((x_start_pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
            ax[1].imshow((pred[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype('uint8'))
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score / n_samples
    return sim_score


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    parser.add_argument("--add-lora-config", type=int, default=0)
    parser.add_argument("--start-from-lora", type=str, default=None)
    parser.add_argument("--dmd", type=int, default=0)
    parser.add_argument("--start-lora-trainable", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
