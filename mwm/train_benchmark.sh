source /etc/network_turbo

# train in benchmark SCAND
python train_sf.py --config config/mwm.yaml \
    --ckpt-every 500 \
    --eval-every 500 \
    --bfloat16 0 \
    --epochs 1  \
    --torch-compile 0 \
    --add-lora-config 1 \
    --log-every 5
