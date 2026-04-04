import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision import transforms

def show_images_with_labelsV2(tensor1, tensor2, tensor3, save_path, file_name):
    start = tensor1.squeeze(0).detach().to(torch.float32)  # [3, H, W]
    preds = tensor2.squeeze(0).detach().to(torch.float32)  # [8, 3, H, W]
    obs = tensor3.detach().to(torch.float32)  # [8, 3, H, W]


    start = (start + 1) / 2
    preds = (preds + 1) / 2
    obs = (obs + 1) / 2

    # 转成 [H, W, C]
    print(start.shape)
    start = start.permute(1, 2, 0).cpu().clamp(0, 1)
    preds = preds.permute(0, 2, 3, 1).cpu().clamp(0, 1)
    obs  = obs.permute(0, 2, 3, 1).cpu().clamp(0, 1)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(3, 8, hspace=0.3)

    # -------- 第一行：start image --------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(start)
    ax.set_title("Start image")
    ax.axis("off")

    # 其余位置空着
    for j in range(1, 8):
        fig.add_subplot(gs[0, j]).axis("off")

    # -------- 第二行：pred images --------
    for i in range(8):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(preds[i])
        ax.axis("off")
        if i == 0:
            #ax.set_ylabel("Pred images", fontsize=12)
            ax.set_title("Pred images")

    # -------- 第三行：observe images --------
    for i in range(8):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(obs[i])
        ax.axis("off")
        if i == 0:
            # ax.set_ylabel("Pred images", fontsize=12)
            ax.set_title("Observe images")

    for j in range(1, 8):
        fig.add_subplot(gs[2, j]).axis("off")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, file_name), bbox_inches="tight")
    plt.close()

def show_images_with_labels(tensor_list, save_path, file_name):
    # tensor_list: [tensor1, tensor2...]
    # tensor1: [N1, 3, H, W], tensor2: [N2, 3, H, W]
    row = len(tensor_list)
    col = max([tensor.shape[0] for tensor in tensor_list])
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(row, col, hspace=0.3)
    for i in range(0, row):
        x = tensor_list[i]
        x = x.detach().to(torch.float32)
        x = (x + 1) / 2
        x = x.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        for j in range(0, col):
            if j >= x.shape[0]:
                fig.add_subplot(gs[i, j]).axis("off")
            else:
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(x[j])
                ax.axis("off")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, file_name), bbox_inches="tight")
    plt.close()

def save_videos(tensor, fps, save_path, file_name):
    # tensor: [N1, 3, H, W]
    import mediapy
    tensor = tensor.detach().to(dtype=torch.float32)
    tensor = (tensor + 1) / 2
    video = tensor.cpu().clamp(0, 1).numpy()  # (T, C, H, W)
    video = video.transpose(0, 2, 3, 1)  # -> (T, H, W, C)

    os.makedirs(save_path, exist_ok=True)
    mediapy.write_video(os.path.join(save_path, file_name), video.astype(np.float32), fps=fps)
