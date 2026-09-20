# RTX 5090 Validation for AMD: MiniMax-H3 with LightX2V

This repository provides the NVIDIA RTX 5090 baseline scripts for the MiniMax-H3
comparison submitted to AMD. It includes configuration generators, AdaLN cache
preparation, BF16 inference, optional INT8 ConvRot inference, and timing extraction.
The AMD/ROCm runs are maintained separately in LightX2VRun.

## Benchmark scope

All supplied inference launchers use 20 steps, 362 generated frames, a configured
size of `[768, 1344]` (height, width), 24 FPS, and seed 42. This is approximately
15 seconds of video. FL2AV uses first/last-frame conditioning; Ref2AV uses a
reference image.

| Precision | Parallelism | GPUs | FL2AV launcher | Ref2AV launcher |
|---|---|---:|---|---|
| BF16 | Sequence parallel (SP2) | 2 | `01_sp2_fl2av.sh` | `02_sp2_ref2av.sh` |
| BF16 | Sequence parallel (SP4) | 4 | `03_sp4_fl2av.sh` | `04_sp4_ref2av.sh` |
| BF16 | Tensor parallel (TP8) | 8 | `07_tp8_fl2av.sh` | `08_tp8_ref2av.sh` |
| INT8 ConvRot | Sequence parallel (SP2) | 2 | — | `09_sp2_int8_ref2av.sh` |
| INT8 ConvRot | Sequence parallel (SP4) | 4 | — | `10_sp4_int8_ref2av.sh` |
| INT8 ConvRot | Sequence parallel (SP8) | 8 | — | `11_sp8_int8_ref2av.sh` |
| INT8 ConvRot | Tensor parallel (TP8) | 8 | — | `12_tp8_int8_ref2av.sh` (optional; not yet validated) |

All launchers are under `scripts/`. The BF16 generator also produces SP8
configurations, but this repository does not include BF16 SP8 launchers.
SP uses Ulysses and block-level CPU offload; TP8 keeps DiT weights on the GPUs.
Both enable text-encoder tensor parallelism and text-encoder CPU offload.
Attention uses `sage_attn2`; AdaLN caching is enabled, feature caching and
compilation are disabled.

## Prerequisites and directory layout

The original test setup used the following resources:

| Component | Requirement |
|---|---|
| GPUs | 8 × NVIDIA RTX 5090, 32 GB each, for the full matrix |
| Host memory | At least 500 GB |
| Free storage | At least 400 GB; allow additional space for retained outputs |
| NVIDIA driver | Version 580 or later, with CUDA 13.0 support |
| Software | Linux, Git, Python 3 with venv support, Docker, NVIDIA Container Toolkit |
| Container image | `lightx2v/lightx2v:26062001-cu130-5090_fix_sage2` |

Run all commands on the GPU host. Replace the `/path/to/...` placeholders below
with absolute paths on your machine. Export these variables in every shell used
to run the scripts; `H3_WORKDIR` is required by the shell launchers.

```bash
export H3_WORKDIR=/path/to/h3-validation
export VALIDATION_REPO=/path/to/5090-Validation-LightX2V
mkdir -p "$H3_WORKDIR"
```

`H3_WORKDIR` holds the source, weights, configurations, caches, logs, and outputs.
`VALIDATION_REPO` is the checkout of this repository. Container paths are fixed
by the launchers.

| Host path | Container path | Purpose |
|---|---|---|
| `$VALIDATION_REPO` | Not mounted | This repository and host launchers |
| `$H3_WORKDIR/LightX2V` | `/workspace/LightX2V` | Upstream inference source |
| `$H3_WORKDIR/MiniMax-H3` | `/data/MiniMax-H3` | Model weights and optional INT8 checkpoint |
| `$H3_WORKDIR/h3_cfg` | `/data/h3_cfg` | Generated configuration files |
| `$H3_WORKDIR/h3_adaln` | `/data/h3_adaln` | Reusable AdaLN caches |
| `$H3_WORKDIR/h3_logs` | `/data/h3_logs` | Cache, conversion, and inference logs |
| `$H3_WORKDIR/h3_out` | `/data/h3_out` | Generated audio/video files |

The container is named `h3_5090`. The commands below mount `$H3_WORKDIR` at `/data`
to match the paths expected by the scripts.

## 1. Prepare source, image, and weights

Check the host, then clone the validation branch and inference source:

```bash
nvidia-smi
df -h "$H3_WORKDIR"
free -g

git clone --branch AMD-Version git@github.com:yangecool/5090-Validation-LightX2V.git \
  "$VALIDATION_REPO"
git clone https://github.com/ModelTC/LightX2V.git "$H3_WORKDIR/LightX2V"
docker pull lightx2v/lightx2v:26062001-cu130-5090_fix_sage2
```

Download the model using an isolated Python environment:

```bash
python3 -m venv "$H3_WORKDIR/.venvs/h3-download"
"$H3_WORKDIR/.venvs/h3-download/bin/pip" install -U modelscope
"$H3_WORKDIR/.venvs/h3-download/bin/modelscope" download \
  --model MiniMax/MiniMax-H3 --local_dir "$H3_WORKDIR/MiniMax-H3"
```

The original checkpoint contains 14 safetensors shards in each of `transformer`,
`transformer_ref`, and `text_encoder`, plus a `vae` directory. Verify the download:

```bash
for component in transformer transformer_ref text_encoder; do
  find "$H3_WORKDIR/MiniMax-H3/$component" -maxdepth 1 -name '*.safetensors' | wc -l
done
ls "$H3_WORKDIR/MiniMax-H3/vae"
```

The LightX2V source revision is not pinned by these scripts. Use the agreed source
revision for a comparison run and record its commit, the model revision, and the
container image digest with the results.

## 2. Start and check the container

```bash
mkdir -p "$H3_WORKDIR/h3_cfg" "$H3_WORKDIR/h3_logs" "$H3_WORKDIR/h3_adaln" "$H3_WORKDIR/h3_out"

docker run -d --name h3_5090 \
  --gpus all \
  --ipc=host --shm-size=64g \
  -v "$H3_WORKDIR:/data" \
  -v "$H3_WORKDIR/LightX2V:/workspace/LightX2V" \
  -w /workspace/LightX2V \
  lightx2v/lightx2v:26062001-cu130-5090_fix_sage2 \
  sleep infinity
```

Check GPU visibility, imports, source, and weights before starting a run:

```bash
docker exec -i h3_5090 python - <<'PY'
import importlib
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPUs:", torch.cuda.device_count())
print("GPU 0 capability:", torch.cuda.get_device_capability(0))
for module in ("sageattention", "flash_attn", "lightx2v_kernel", "sgl_kernel",
               "diffusers", "transformers"):
    importlib.import_module(module)
    print("OK", module)
PY

docker exec h3_5090 test -f /workspace/LightX2V/lightx2v/infer.py
docker exec h3_5090 ls /data/MiniMax-H3/transformer
```

## 3. Generate configurations and AdaLN caches

```bash
cd "$VALIDATION_REPO"
python3 scripts/gen_configs.py "$H3_WORKDIR/h3_cfg"
bash scripts/00_cache_adaln.sh
```

The cache script launches separate FL2AV and Ref2AV jobs, waits a fixed interval,
and lists `$H3_WORKDIR/h3_adaln/minimax_h3/`. It does not wait for confirmed job completion.
Inspect both `cache_*.log` files in `$H3_WORKDIR/h3_logs` and verify successful cache
creation before inference. Reuse these caches across GPU counts for the same
model and generation settings; regenerate them when those inputs change.

## 4. Run the BF16 baseline

Choose a launcher from the matrix and run it from this repository. For example:

```bash
bash scripts/07_tp8_fl2av.sh
```

The other BF16 cases are:

```bash
bash scripts/01_sp2_fl2av.sh
bash scripts/02_sp2_ref2av.sh
bash scripts/03_sp4_fl2av.sh
bash scripts/04_sp4_ref2av.sh
bash scripts/08_tp8_ref2av.sh
```

Run each command separately. Each launcher starts inference in a detached
container process and follows its log with `tail -f`. Wait for successful output
saving and final timing lines, then press Ctrl+C to stop following the log before
starting the next case. Ctrl+C does not stop the detached inference process.
Launchers select GPU IDs starting at 0; avoid overlapping runs.

FL2AV uses `flf2v_input_first_frame-fs8.png` and
`flf2v_input_last_frame-fs8.png` from the upstream `assets/inputs/imgs` directory.
Ref2AV selects the first `*.jpg` returned from that directory and prints its path.
Confirm the selected files before comparing results.

## 5. Run the optional INT8 ConvRot cases

Convert the Ref2AV DiT weights on the CPU, then generate the INT8 configurations:

```bash
docker exec h3_5090 pip install "comfy-kitchen>=0.2.15"
bash scripts/convert_h3_int8_convrot.sh
```

The converter also runs detached and follows its log. Wait for conversion to
finish successfully, then press Ctrl+C. Its output is
`$H3_WORKDIR/MiniMax-H3/h3_int8/minimax_h3_ref2va_int8_convrot.safetensors`.
Do not run inference during conversion.

```bash
python3 scripts/gen_int8_configs.py "$H3_WORKDIR/h3_cfg" "$H3_WORKDIR/h3_cfg"
```

Run the desired cases individually using the same log-completion procedure:

```bash
bash scripts/09_sp2_int8_ref2av.sh
bash scripts/10_sp4_int8_ref2av.sh
bash scripts/11_sp8_int8_ref2av.sh

# Optional TP8 comparison; this case has not yet been validated.
bash scripts/12_tp8_int8_ref2av.sh
```

## 6. Collect results and compare with LightX2VRun

Logs are timestamped as `h3_logs/run_5090_<case>_<timestamp>.log`.
Videos are saved as `h3_out/5090_<case>_362f.mp4`; rerunning a case overwrites
its video, so preserve outputs before repeating it.

Extract timings from the latest log, or process all logs one at a time:

```bash
bash scripts/extract.sh

for log in "$H3_WORKDIR"/h3_logs/run_5090_*.log; do
  [ -f "$log" ] || continue
  bash scripts/extract.sh "$log"
done
```

The extractor reads Rank 0 stage timings, including model loading, encoding,
DiT, decoding, output saving, pipeline time, and total time, plus per-step DiT
values. Missing fields appear as `-` or empty values. It accepts one log path per
invocation. The memory line is available only when the runtime emits it; capture
`nvidia-smi` during inference for TP8 memory measurements.

For each submission, retain the raw log, generated JSON configuration, output
video, GPU/driver details, source commits, image digest, and model/input revisions.
Report model loading, DiT, pipeline, and total times separately, and identify
cold-cache versus warm-cache runs.

Recent LightX2VRun changes separate 5-second and 15-second BF16 FL2AV profiles,
bundle first/last-frame assets and prompts, and prepare reusable AdaLN caches.
When pairing this baseline with an AMD run:

- Match model variant, checkpoint, precision, sampling settings, generated frame
  count, resolution, FPS, seed, GPU count, and cache state.
- Use identical input images and prompts. These launchers use upstream sample
  images and short prompts; they do not automatically use LightX2VRun's umbrella
  keyframes, motion prompt, or Ref2AV reference profiles.
- Record output-duration handling. LightX2VRun's 15-second FL2AV profile generates
  362 frames and trims to 360; these launchers request 362 frames without setting
  an explicit output-duration trim.
- Record parallelism, offload, and backend differences. Compare equivalent timing
  stages rather than assuming the two repositories produce identical summaries.

The supplied commands reproduce this repository's NVIDIA workload. Matching a
specific LightX2VRun profile requires aligning its inputs and settings before
collecting comparison measurements.
