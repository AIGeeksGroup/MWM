# MWM: Mobile World Models for Action-Conditioned Consistent Prediction

This is the official repository for the paper:
> **MWM: Mobile World Models for Action-Conditioned Consistent Prediction**
>
> Han Yan\*, Zishang Xiang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, and Hao Tang<sup>‡</sup>
>
> School of Computer Science, Peking University
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author
>
> ### [Paper](https://arxiv.org/abs/2603.07799v1) | [Website](https://aigeeksgroup.github.io/MWM/) | [Model](https://huggingface.co/AIGeeksGroup/MWM)

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{yan2026mwm,
  title={MWM: Mobile World Models for Action-Conditioned Consistent Prediction},
  author={Yan, Han and Xiang, Zishang and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2603.07799},
  year={2026}
}
```
---

## 🏃 Intro MWM
World models enable planning in imagined future
predicted space, offering a promising framework for embodied
navigation. However, existing navigation world models often
lack action-conditioned consistency, so visually plausible pre-
dictions can still drift under multi-step rollout and degrade
planning. Moreover, efficient deployment requires few-step dif-
fusion inference, but existing distillation methods do not explic-
itly preserve rollout consistency, creating a training–inference
mismatch. To address these challenges, we propose MWM, a
mobile world model for planning-based image-goal navigation.
Specifically, we introduce a two-stage training framework
that combines structure pretraining with Action-Conditioned
Consistency (ACC) post-training to improve action-conditioned
rollout consistency. We further introduce Inference-Consistent
State Distillation (ICSD) for few-step diffusion distillation with
improved rollout consistency. Our experiments on benchmark
and real-world tasks demonstrate consistent gains in visual
fidelity, trajectory accuracy, planning success, and inference
efficiency.

## 📰 News
<b>2026/03/12:</b> 🎉 Our paper has been promoted by <a href="https://www.xiaohongshu.com/explore/69b0c5fd000000000c00a074?note_flow_source=wechat&xsec_token=CBjkQExtIAPGVW65vEg7ih39TWxQI9Ar8bqORtOf0xNE4="><b>Heart of Embodied Intelligence</b></a>.

## TODO List

- [x] Upload our paper to arXiv and build project pages.
- [x] Upload the code.
- [ ] Upload the model.


## ⚡ Quick Start
### Environment Setup

Clone the repository and Create a conda environment:
```bash
git clone https://github.com/AIGeeksGroup/MWM.git
cd MWM
conda create -n mwm python=3.10
conda activate mwm
pip install -r requirements.txt
```

### Data
Please follow the official download and preprocess guide at [NWM](https://github.com/facebookresearch/nwm) for detailed data download and preprocessing instructions.

### Training

Two-stage training (Structure Pretraining + Action-Conditioned Consistency (ACC) Post-training)
```bash
cd mwm
bash finetune_in_envs.sh
```

### Evaluation
Evaluate ACC and Generation Quality in SCAND
```bash
bash single_frame_evaluation.sh
```

Evaluate Navigation Performance in SCAND
```bash
bash trajectory_evaluation.sh
```

### Deployment in MMK2
```bash
cd realworld_deploy
```

#### Server
Start the Inference Service
```bash
bash start_nwm_infer_service.sh
```

#### Client 
The client connects to both the MMK2 robot and the inference server
```bash
cd policies/nwm/real
```

Collect data with MMK2
```bash
python record_data.py
```

Data Processing
```bash
python process_episodes.py
```

Start the Client
```bash
ssh -p <SSH_PORT> -L 8000:127.0.0.1:8000 <USERNAME>@<SERVER_HOST>
```
---


## 😘 Acknowledgement
We thank the authors of [NWM](https://github.com/facebookresearch/nwm) for their open-source code. 