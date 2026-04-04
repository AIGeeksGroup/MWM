source /etc/network_turbo

# real beike
export NUM_NODES=1
export HOST_NODE_ADDR=127.0.0.1
export CURR_NODE_RANK=0

# stage1: teacher-forcing finetune
torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=4 \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${HOST_NODE_ADDR}:29500 \
  train.py --config config/mwm_real_tf_pretrain.yaml \
    --ckpt-every 500 \
    --eval-every 1000 \
    --bfloat16 0 \
    --epochs 595  \
    --torch-compile 0 \
    --log-every 10

# stage2: self-forcing finetune
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc-per-node=4 \
    --node-rank=${CURR_NODE_RANK} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${HOST_NODE_ADDR}:29500 \
    train_sf.py --config config/mwm_real_sf_posttrain_adaLN.yaml \
        --ckpt-every 200 \
        --eval-every 200 \
        --bfloat16 0 \
        --epochs 20  \
        --torch-compile 0 \
        --add-lora-config 1 \
        --log-every 1 
