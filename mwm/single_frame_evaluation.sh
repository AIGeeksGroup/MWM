source /etc/network_turbo
RESULTS_FOLDER=./single_frame_evaluation_output

mode='mwm_real_sf_posttrain_adaLN'
datasets=real_beike

if [[ "$mode" == "baseline" ]]; then
  echo "Running in MODE=baseline"
  exp=nwm_cdit_xl
  diffusion_steps=25
  ckp=cdit_xl_100000
  lora_dir=null
  eval_type=rollout

elif [[ "$mode" == "ego4d" ]]; then
  echo "Running in MODE=ego4d"
  exp=nwm_cdit_xl
  diffusion_steps=25
  ckp=cdit_xl_ego4d_200000
  lora_dir=null
  eval_type=rollout

elif [[ "$mode" == "mwm" ]]; then
  echo "Running in MODE=mwm"
  exp=mwm
  diffusion_steps=5
  ckp=cdit_xl_100000
  lora_dir=logs/mwm/checkpoints/lora_100500
  eval_type=rollout

elif [[ "$mode" == "mwm_real_tf_pretrain" ]]; then
  exp=mwm_real_tf_pretrain
  diffusion_steps=25
  ckp=0126000
  lora_dir=null
  eval_type=rollout

elif [[ "$mode" == "mwm_real_sf_posttrain" ]]; then
  exp=mwm_real_sf_posttrain
  diffusion_steps=5
  ckp=0126000
  lora_dir=logs/mwm_real_sf_posttrain/checkpoints/lora_127100
  eval_type=rollout
 
elif [[ "$mode" == "mwm_real_sf_posttrain_adaLN" ]]; then
  exp=mwm_real_sf_posttrain_adaLN
  diffusion_steps=5
  ckp=0126000
  eval_type=rollout
  lora_dir=logs/mwm_real_sf_posttrain_adaLN/checkpoints/lora_126400

else
  echo "ERROR: unknown MODE='$MODE'. Must be one of: mode | baseline | ego4d" >&2
  exit 1
fi

# Prepare ground truth frames for evaluation (one-time)
python isolated_nwm_infer.py \
    --exp config/${exp}.yaml \
    --datasets ${datasets} \
    --batch_size 40 \
    --num_workers 12 \
    --eval_type ${eval_type} \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1

# Predict future state given action
python isolated_nwm_infer.py \
    --exp config/${exp}.yaml \
    --ckp ${ckp} \
    --datasets ${datasets} \
    --diffusion_steps ${diffusion_steps} \
    --batch_size 30 \
    --num_workers 12 \
    --eval_type ${eval_type} \
    --output_dir ${RESULTS_FOLDER} \
    --lora-dir ${lora_dir} 

# Report metrics compared to GT (LPIPS, DreamSim, FID)
python isolated_nwm_eval.py \
    --datasets ${datasets} \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/${exp}_${ckp} \
    --eval_types ${eval_type} \
    --batch_size 2 
