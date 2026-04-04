source /etc/network_turbo
cd policies/nwm
python nwm_infer_service.py \
     --config /root/autodl-tmp/embodiedAI/DISCOVERSE/policies/nwm/config/real_infer_sf.yaml \
     --port 8000 \
     --output-dir server_outputs
