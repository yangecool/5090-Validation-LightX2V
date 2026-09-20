#!/bin/bash
# MiniMax-H3 RTX 5090 inference: sp4_fl2av.
# 20 steps, 362 frames, 768 x 1344, 24 FPS, seed 42.
set -u

: "${H3_WORKDIR:?Set H3_WORKDIR to the absolute host workspace path mounted at /data}"

TAG=sp4_fl2av
TS=$(date +%m%d_%H%M%S)
LOG=/data/h3_logs/run_5090_${TAG}_${TS}.log

docker exec -d h3_5090 bash -lc "
cd /workspace/LightX2V
export CUDA_VISIBLE_DEVICES=0,1,2,3
export lightx2v_path=/workspace/LightX2V model_path=/data/MiniMax-H3
source scripts/base/base.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \\
  --model_cls minimax_h3 --model-variant fl2av --task fl2av \\
  --model_path /data/MiniMax-H3 \\
  --config_json /data/h3_cfg/5090_sp4_20step_768p_362f_fl2av.json \\
  --prompt 'Create a coherent transition with natural synchronized sound.' \\
  --image_path /workspace/LightX2V/assets/inputs/imgs/flf2v_input_first_frame-fs8.png \\
  --last_frame_path /workspace/LightX2V/assets/inputs/imgs/flf2v_input_last_frame-fs8.png \\
  --save_result_path /data/h3_out/5090_${TAG}_362f.mp4 --seed 42 \\
  > ${LOG} 2>&1"

sleep 20 && tail -f "${H3_WORKDIR}/h3_logs/run_5090_${TAG}_${TS}.log"
