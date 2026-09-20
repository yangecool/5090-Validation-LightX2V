#!/bin/bash
# Extract Rank 0 timings from one log, or the latest workspace log.
# Usage: bash scripts/extract.sh [LOG_FILE]
set -u

: "${H3_WORKDIR:?Set H3_WORKDIR to the absolute host workspace path mounted at /data}"

if [ "${1:-}" != "" ] && [ -f "${1:-}" ]; then
  L="$1"
else
  L=$(ls -t "${H3_WORKDIR}"/h3_logs/run_5090_*.log 2>/dev/null | head -1)
fi
[ -n "${L:-}" ] && [ -f "$L" ] || { echo "❌ 找不到日志（用法: bash extract.sh <日志文件>）"; exit 1; }

echo "=== $(basename "$L") ==="
for k in "Load models" "Run Text Encoder" "Run VAE Encoder" "Run Input Encoder" \
         "Run DiT" "Offload DiT" "Run Video VAE Decoder" "Run Audio VAE Decoder" \
         "Run VAE Decoder" "Save Audio-Video Output" "RUN pipeline" "Total Cost"; do
  v=$(grep -m1 "Rank 0 - .*${k} cost" "$L" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  printf "%-26s %s\n" "$k" "${v:--}"
done
echo "packed layout: $(grep -m1 'packed layout' "$L" | sed 's/.*packed layout: //')"
echo "显存:          $(grep -m1 'Emptying device cache' "$L" | sed 's/.*\[Memory\] //')"
echo "成片:          $(grep -m1 'output saved to' "$L" | sed 's/.*output saved to //')"
echo "单步:          $(grep 'Rank 0 -' "$L" | grep -oE 'Run Dit every step cost [0-9.]+' | awk '{print $6}' | paste -sd' ')"
