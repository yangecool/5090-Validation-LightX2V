#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 5090 上 MiniMax-H3 的全部 8 份运行配置（SP2/SP4/SP8 × fl2av/ref2av + TP8 × fl2av/ref2av）。

统一规格：20 步 / 362 帧 / 768x1344 / 24fps / seed 42（= 15 秒视频）

用法：
    # 在本机（Windows/Linux 任意有 python3 的环境）
    python gen_configs.py <输出目录>

    # 在 5090 宿主上（写进 $H3_WORKDIR/h3_cfg，容器内可见为 /data/h3_cfg）
    python3 gen_configs.py $H3_WORKDIR/h3_cfg

关键字段说明：
    parallel                       SP 用 {"seq_p_size": N, "seq_p_attn_type": "ulysses"}
                                   TP 用 {"tensor_p_size": 8}
    text_encoder_tensor_parallel   设为 true（TE 按卡数切分，每卡只装 1/N）
    cpu_offload                    SP 档为 true（权重按需搬运）；TP8 档为 false（权重常驻显存）
    text_encoder_cpu_offload       true（TE 整个流程只用一次）
"""
import json
import os
import sys


def cfg(task, parallel, offload=True, te_offload=True):
    d = {
        "infer_steps": 20,
        "num_frames": 362,
        "size": [768, 1344],
        "fps": 24,
        "enable_cfg": False,
        "cpu_offload": offload,
        "offload_granularity": "block",
        "text_encoder_cpu_offload": te_offload,
        "vae_cpu_offload": True,
        "vae_decode_parallel": True,
        "use_adaln_cache": True,
        "adaln_cache_dir": "/data/h3_adaln",
        "lazy_load": False,
        "unload_modules": False,
        "attn_type": "sage_attn2",
        "rms_type": "sgl-kernel",
        "rope_type": "minimax_h3_triton_rope",
        "feature_caching": "NoCaching",
        "use_compile": False,
        "video_flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "vae_spatial_scale_factor": 16,
        "audio_sampling_rate": 32000,
        "text_encoder_tensor_parallel": True,   # TE 按卡数切分（每卡 1/N）
        "parallel": parallel,
    }
    if task == "ref2av":
        d["reference_image_resize_mode"] = "match"
    return d


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 scripts/gen_configs.py OUTPUT_DIR")
    out = os.path.abspath(os.path.expanduser(sys.argv[1]))
    os.makedirs(out, exist_ok=True)
    written = []

    # SP 系列（序列并行）：权重按需搬运
    for n in (2, 4, 8):
        for task in ("fl2av", "ref2av"):
            p = os.path.join(out, f"5090_sp{n}_20step_768p_362f_{task}.json")
            d = cfg(task, {"seq_p_size": n, "seq_p_attn_type": "ulysses"}, True, True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
            written.append(p)

    # TP8（张量并行）：DiT 常驻显存，TE 卸载
    for task in ("fl2av", "ref2av"):
        p = os.path.join(out, f"5090_tp8_20step_768p_362f_{task}.json")
        d = cfg(task, {"tensor_p_size": 8}, False, True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
        written.append(p)

    for p in written:
        print("wrote", p)
    print(f"\n共 {len(written)} 份配置 -> {out}")


if __name__ == "__main__":
    main()
