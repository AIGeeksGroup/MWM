source /etc/network_turbo
torchrun --nproc-per-node=6 planning_eval.py \
    --exp config/mwm.yaml   \
    --datasets scand   \
    --rollout_stride 1   \
    --batch_size 1   \
    --num_samples 120   \
    --topk 5   \
    --num_workers 12   \
    --output_dir ./trajectory_evaluation_output   \
    --save_preds   \
    --ckp cdit_xl_100000   \
    --opt_steps 1   \
    --num_repeat_eval 3 \
    --lora-dir logs/mwm/checkpoints/lora_137000
