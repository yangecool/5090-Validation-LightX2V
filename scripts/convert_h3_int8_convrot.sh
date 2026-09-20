#!/bin/bash
# Convert Ref2AV BF16 weights to INT8 ConvRot using the CPU.
# Requires comfy-kitchen in the container.
set -u

: "${H3_WORKDIR:?Set H3_WORKDIR to the absolute host workspace path mounted at /data}"

# 改成 transformer（fl2av 的 base 权重）即可转 fl2va 版本
SRC=/data/MiniMax-H3/transformer_ref
OUT_DIR=/data/MiniMax-H3/h3_int8
OUT_NAME=minimax_h3_ref2va_int8_convrot

TLOG=/data/h3_logs/convert_int8_ref2av_$(date +%m%d_%H%M%S).log

mkdir -p "${H3_WORKDIR}/MiniMax-H3/h3_int8" 2>/dev/null || true

docker exec -d h3_5090 bash -lc "
cd /workspace/LightX2V
export lightx2v_path=/workspace/LightX2V model_path=/data/MiniMax-H3
source scripts/base/base.sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
python3 tools/convert/converter.py \
  --source ${SRC} \
  --output ${OUT_DIR} \
  --output_name ${OUT_NAME} \
  --output_ext .safetensors \
  --model_type h3 \
  --device cpu \
  --quantized \
  --bits 8 \
  --linear_type int8-convrot \
  --single_file \
  > ${TLOG} 2>&1"

sleep 20 && tail -f "${H3_WORKDIR}"/h3_logs/convert_int8_ref2av_*.log
