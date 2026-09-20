#!/bin/bash
# Prepare FL2AV and Ref2AV AdaLN caches before inference.
set -u

: "${H3_WORKDIR:?Set H3_WORKDIR to the absolute host workspace path mounted at /data}"

C=h3_5090

for V in fl2av ref2av; do
  echo "== 生成 ${V} 的 AdaLN cache =="
  docker exec -d $C bash -lc "
cd /workspace/LightX2V
export lightx2v_path=/workspace/LightX2V model_path=/data/MiniMax-H3
source scripts/base/base.sh
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path /data/MiniMax-H3 \
  --config_json /data/h3_cfg/5090_sp4_20step_768p_362f_${V}.json \
  --model-variant ${V} \
  > /data/h3_logs/cache_${V}_\$(date +%m%d_%H%M%S).log 2>&1"
  sleep 5
done

echo "等待生成（约 1 分钟）..."
sleep 60
echo "== 产物 =="
ls -R "${H3_WORKDIR}/h3_adaln/minimax_h3/"
