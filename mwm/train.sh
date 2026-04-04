source /etc/network_turbo
python train_sf.py --config config/mwm.yaml --ckpt-every 500 --eval-every 500 --bfloat16 0 --epochs 1  --torch-compile 0 --peft-lora-finetune 1 --log-every 5
