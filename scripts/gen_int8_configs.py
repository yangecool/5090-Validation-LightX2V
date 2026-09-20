#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 bf16 配置基础上生成 int8-convrot 版配置（只加三个字段，其它参数不动）。

用法:
    # 远端部署：从 $H3_WORKDIR/h3_cfg 的 bf16 配置生成 int8 版（输出到同目录）
    python3 gen_int8_configs.py $H3_WORKDIR/h3_cfg $H3_WORKDIR/h3_cfg

    # 本地仓库（从 configs/ 的 bf16 版生成 int8 版）
    python3 gen_int8_configs.py <repo>/configs <repo>/configs

生成 4 份：5090_{sp2,sp4,sp8,tp8}_int8_20step_768p_362f_ref2av.json
"""
import json
import os
import sys

CKPT = "/data/MiniMax-H3/h3_int8/minimax_h3_ref2va_int8_convrot.safetensors"
QUANT = {
    "dit_quantized": True,
    "dit_quant_scheme": "int8-convrot",
    "dit_quantized_ckpt": CKPT,   # 文件路径（不是目录）
}


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python3 scripts/gen_int8_configs.py SOURCE_DIR [OUTPUT_DIR]")
    src = os.path.abspath(os.path.expanduser(sys.argv[1]))
    dst = os.path.abspath(os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else src))
    os.makedirs(dst, exist_ok=True)
    n = 0
    for tag in ("sp2", "sp4", "sp8", "tp8"):
        p_in = os.path.join(src, f"5090_{tag}_20step_768p_362f_ref2av.json")
        if not os.path.isfile(p_in):
            print("skip (not found):", p_in)
            continue
        with open(p_in, encoding="utf-8") as f:
            d = json.load(f)
        d.update(QUANT)
        p_out = os.path.join(dst, f"5090_{tag}_int8_20step_768p_362f_ref2av.json")
        with open(p_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
        print("wrote", p_out, "|", d.get("parallel"), "| cpu_offload =", d.get("cpu_offload"))
        n += 1
    print(f"\n共 {n} 份 int8 配置；dit_quantized_ckpt = {CKPT}")


if __name__ == "__main__":
    main()
