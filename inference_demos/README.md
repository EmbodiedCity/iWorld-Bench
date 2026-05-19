# Inference Demo Scripts

This folder contains dry-run inference adapters for public iWorldBench verification. These scripts do not execute real inference. They validate CSV rows, first-frame paths, task filters, controls, and model-specific command/config construction. Most adapters consume camera-control TXT files; MatrixGame-2 consumes keyboard/mouse schedules, text-prompt adapters consume prompts derived from the same metadata, and HY-WorldPlay consumes original source-video/source-camera trajectory pairs for CameraFollowing-style inference.

## Task modes

The release metadata contains three task modes:

| Mode | CSV value | Control files | Recommended evaluation |
|---|---|---|---|
| Action-control / difficulty tasks | `Diff` | `camera_<level>_<translation>_<rotation>.txt` | `--metric action_control` |
| Memory loop-closure tasks | `Mem` | `memory_<id>.txt` | `--metric memory_ability` |
| Camera-following / original trajectory tasks | `CameraFollowing` | `source_camera_txt/*.txt` | `--metric camera_following` |

Keep generated outputs for `Diff` and `Mem` in separate directories when computing task-specific metrics.

## Included model demos

These adapters reference external projects but do not vendor their model code or checkpoints. If you use an adapter, reproduce a baseline, or compare against one of these models, please cite and follow the license terms of the original repository.

| Script | Task support | Control mode | Original repository |
|---|---|---|---|
| `run_cami2v_demo.py` | `Diff`, `Mem` | camera TXT | [CamI2V](https://github.com/ZGCTroy/CamI2V) |
| `run_motionctrl_demo.py` | `Diff`, `Mem` | camera TXT | [MotionCtrl](https://github.com/TencentARC/MotionCtrl) |
| `run_cameractrl_demo.py` | `Diff`, `Mem` | camera TXT | [CameraCtrl](https://github.com/hehao13/CameraCtrl) |
| `run_realcami2v_demo.py` | `Diff`, `Mem` | camera TXT | [RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V) |
| `run_videox_demo.py` | `Diff`, `Mem` | camera TXT | [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) |
| `run_ac3d_demo.py` | `Diff`, `Mem` | camera TXT | [AC3D](https://github.com/snap-research/ac3d) |
| `run_matrixgame_demo.py` | `Diff`, `Mem` | keyboard/mouse schedule | [Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) |
| `run_wan_demo.py` | `Diff`, `Mem` | text prompt | [Wan2.2](https://github.com/Wan-Video/Wan2.2) |
| `run_yume_demo.py` | `Diff`, `Mem` | text prompt | [YUME](https://github.com/stdstu12/YUME) |
| `run_cogvideox_demo.py` | `Diff`, `Mem` | text prompt | [CogVideo](https://github.com/THUDM/CogVideo) |
| `run_hunyuan_demo.py` | `Diff`, `Mem` | text prompt | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) |
| `run_hyworldplay_demo.py` | `CameraFollowing` | source video + source camera TXT directories | [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) |

MatrixGame-2 is different from the camera-TXT adapters. Its dry-run builds keyboard/mouse control schedules: `Diff` uses a single repeated keyboard one-hot vector plus mouse delta from `translation` and `rotation`; `Mem` uses `frame 0` static, `frames 1-40` for the first memory action, and `frames 41-80` for the second memory action.

The text-prompt adapters convert `Diff` action metadata or `Mem` memory IDs into English camera-motion prompts. Wan2.2, CogVideoX, and HunyuanVideo-1.5 command previews use a generated two-column prompt CSV with `Image Filename` and `Prompt`. YUME command previews target its action and memory wrapper scripts, so the dry-run also prepares YUME-specific action or memory adapter CSVs from the same metadata. HY-WorldPlay is directory-level: it previews a command that passes source videos via `--videos_dir` and original source-camera TXT files via `--cameras_dir`. Its external runner pairs file stems after replacing `_video_` and `_camera_` separators; if your source-video names contain `_camera_` as ordinary text, prepare compatible symlinks or adapt the external runner's pairing logic.

## Quick dry-run

From the repository root:

```bash
python3 inference_demos/run_videox_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --cameras-dir camera_trajectories/inference_txt \
  --tasks Diff \
  --max-samples 3
```

Memory-mode validation:

```bash
python3 inference_demos/run_ac3d_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --cameras-dir camera_trajectories/inference_txt \
  --tasks Mem \
  --max-samples 3
```

MatrixGame action validation:

```bash
python3 inference_demos/run_matrixgame_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --tasks Diff \
  --levels 1 2 \
  --max-samples 3
```

MatrixGame memory validation:

```bash
python3 inference_demos/run_matrixgame_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --tasks Mem \
  --max-samples 3
```

Text-prompt validation:

```bash
python3 inference_demos/run_wan_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --tasks Diff \
  --max-samples 3
```

```bash
python3 inference_demos/run_yume_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --tasks Mem \
  --max-samples 3
```

```bash
python3 inference_demos/run_cogvideox_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --tasks Diff \
  --max-samples 3
```

```bash
python3 inference_demos/run_hunyuan_demo.py \
  --csv dataset/all_pack/metadata.csv \
  --assets-root dataset/all_pack \
  --tasks Diff \
  --max-samples 3
```

CameraFollowing / original trajectory validation:

```bash
python3 inference_demos/run_hyworldplay_demo.py \
  --csv dataset/all_pack/camera_following_metadata.csv \
  --assets-root dataset/all_pack \
  --cameras-dir camera_trajectories/source_camera_txt \
  --source-videos-dir /path/to/source_videos \
  --max-samples 3
```

## Optional model path checks

By default, model checkpoints and external project roots are treated as placeholders so the dry-run can pass in a clean release environment. To verify local model paths too, pass `--check-model-paths` and provide the corresponding paths or environment variables.

Examples:

```bash
VIDEOXFUN_ROOT=/path/to/VideoX-Fun \
VIDEOXFUN_MODEL_PATH=/path/to/model \
python3 inference_demos/run_videox_demo.py --tasks Diff --max-samples 3 --check-model-paths
```

```bash
MATRIXGAME_ADAPTER_ROOT=/path/to/VideoX-Fun/evaluation_system \
MATRIXGAME_CHECKPOINT=/path/to/base_distill.safetensors \
MATRIXGAME_CONFIG_PATH=/path/to/inference_universal.yaml \
MATRIXGAME_PRETRAINED_MODEL_PATH=/path/to/Matrix-Game-2.0 \
python3 inference_demos/run_matrixgame_demo.py --tasks Mem --max-samples 3 --check-model-paths
```

```bash
WAN_ADAPTER_ROOT=/path/to/wan/adapter \
WAN_PYTHON=/path/to/python \
python3 inference_demos/run_wan_demo.py --tasks Diff --max-samples 3 --check-model-paths
```

```bash
HYWORLDPLAY_ROOT=/path/to/HY-WorldPlay \
HYWORLDPLAY_PYTHON=/path/to/worldplay/bin/python \
HYWORLDPLAY_MODEL_PATH=/path/to/HunyuanVideo-1.5 \
HYWORLDPLAY_ACTION_CKPT=/path/to/ar_distilled_action_model/model.safetensors \
HYWORLDPLAY_SOURCE_VIDEOS_DIR=/path/to/source_videos \
python3 inference_demos/run_hyworldplay_demo.py --max-samples 3 --check-model-paths
```

## Dry-run output

Each script prints:

- Validation counts for total, filtered, missing, invalid, and valid CSV rows.
- Task/control distribution.
- A model config preview.
- Example constructed commands.
- Text prompt rows for text-prompt adapters.

Use `--write-manifest /path/to/manifest.json` to save the full validated sample list and command list. For text-prompt adapters, this also writes generated CSV inputs under the output directory. Wan2.2, CogVideoX, and HunyuanVideo-1.5 use the two-column prompt CSV directly; YUME uses generated action or memory adapter CSVs in its constructed commands. HY-WorldPlay uses the packaged `camera_following_metadata.csv` only for validation and command preview; its external runner performs directory-level pairing between `source_video_filename` and `source_camera_txt_path` stems.
