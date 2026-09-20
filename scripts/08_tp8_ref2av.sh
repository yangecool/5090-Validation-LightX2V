#!/bin/bash
# MiniMax-H3 RTX 5090 inference: tp8_ref2av.
# 20 steps, 362 frames, 768 x 1344, 24 FPS, seed 42.
set -u

: "${H3_WORKDIR:?Set H3_WORKDIR to the absolute host workspace path mounted at /data}"

TAG=tp8_ref2av
TS=$(date +%m%d_%H%M%S)
LOG=/data/h3_logs/run_5090_${TAG}_${TS}.log

IMG=$(docker exec -t h3_5090 bash -lc 'ls /workspace/LightX2V/assets/inputs/imgs/*.jpg 2>/dev/null | head -1' | tr -d '\r')
echo "参考图: $IMG"

docker exec -d h3_5090 bash -lc "
cd /workspace/LightX2V
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export lightx2v_path=/workspace/LightX2V model_path=/data/MiniMax-H3
source scripts/base/base.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \\
  --model_cls minimax_h3 --model-variant ref2av --task ref2av \\
  --model_path /data/MiniMax-H3 \\
  --config_json /data/h3_cfg/5090_tp8_20step_768p_362f_ref2av.json \\
  --prompt 'Generate an audio-video scene following the references.' \\
  --image_path ${IMG} \\
  --save_result_path /data/h3_out/5090_${TAG}_362f.mp4 --seed 42 \\
  > ${LOG} 2>&1"

sleep 20 && tail -f "${H3_WORKDIR}/h3_logs/run_5090_${TAG}_${TS}.log"
