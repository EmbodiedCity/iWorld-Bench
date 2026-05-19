# iWorldBench Video Metrics — Usage Guide

## Overview

The unified entry point is `unified_video_metrics.py` at the root of this repository.

Most functions share the same signature:
```
function(video_dir, save_dir, max_workers=4)
```

- **`video_dir`**: directory containing generated `.mp4` videos (scanned recursively).
- **`save_dir`**: directory where CSV result files are written (created automatically).
- **`max_workers`**: thread count for parallel processing (default 4).
- VBench functions additionally support `gpu`, `overwrite`, `retry_failed`, and `limit`.

The paper reports three metric groups with 9 metrics:

| Category | Metrics | GPU Required |
|---|---|---|
| **Generation Quality** | Image Quality, Brightness Consistency, Color Temperature Constraint, Sharpness Retention | Image Quality requires VBench |
| **Trajectory Following** | Motion Smoothness, Trajectory Accuracy, Trajectory Tolerance | Yes, VBench + VIPe |
| **Memory Ability** | Memory Symmetry, Trajectory Alignment | Trajectory Alignment requires VIPe |

The packaged metadata has two standard task modes: `Diff` for action-control/difficulty tasks and `Mem` for memory loop-closure tasks. It also includes `camera_following_metadata.csv` for optional `CameraFollowing` evaluation, which is only for models that take explicit camera trajectories as input. For task-specific evaluation, keep generated videos in separate directories and run `--metric action_control` for `Diff`, `--metric memory_ability` for `Mem`, and `--metric camera_following` for trajectory-input models. `noise`, `video_quality`, `trajectory`, and `vbench` remain available as compatibility/diagnostic bundles, but they are not the paper's main taxonomy.

---

## Quick Start — Python API

```python
import sys
sys.path.insert(0, "/path/to/iworld-bench")  # repo root
from unified_video_metrics import (
    calculate_brightness_consistency,
    calculate_color_temperature,
    calculate_video_noise,
    calculate_sharpness_retention,
    calculate_memory_symmetry,
    calculate_trajectory_accuracy,
    calculate_trajectory_alignment,
    calculate_trajectory_tolerance,
    calculate_imaging_quality,
    calculate_motion_smoothness,
    calculate_generation_quality,
    calculate_trajectory_following,
    calculate_memory_ability,
    calculate_action_control,
    calculate_camera_following,
    calculate_all_video_quality,
    calculate_all_trajectory,
    calculate_all_vbench,
    calculate_all,
)

VIDEO_DIR = "/path/to/your/generated_videos"
SAVE_DIR  = "/path/to/results"

# Run Generation Quality from the paper
calculate_generation_quality(
    VIDEO_DIR,
    SAVE_DIR,
    max_workers=4,
    vbench_gpu="0",
)

# Run Trajectory Following from the paper
calculate_trajectory_following(
    VIDEO_DIR,
    SAVE_DIR,
    source_npz_dir="./camera_trajectories/reference_npz",
    camera_txt_dir="./camera_trajectories/inference_txt",
    vbench_gpu="0",
)

# Run Memory Ability from the paper
calculate_memory_ability(VIDEO_DIR, SAVE_DIR, max_workers=4)

# Run Camera Following for trajectory-input models only
calculate_camera_following(
    VIDEO_DIR,
    SAVE_DIR,
    source_npz_dir="./camera_trajectories/source_reference_npz",
    vbench_gpu="0",
)

# Run all 9 paper metrics
calculate_all(
    VIDEO_DIR,
    SAVE_DIR,
    source_npz_dir="./camera_trajectories/reference_npz",
    camera_txt_dir="./camera_trajectories/inference_txt",
)
```

CSV reports are written to `{SAVE_DIR}/reports/`.

---

## Environment Setup

### Required Paths

All examples below reference these four placeholders. Export them once, or
replace them with your own local paths:

```bash
# Root of THIS repository (contains unified_video_metrics.py)
export IWORLD_BENCH_ROOT=/path/to/iworld-bench

# Your conda environment that has Python + the dependencies installed.
# We recommend the "vipe" env; see VIPe's official docs for setup.
export VIPE_ENV=/path/to/your/conda_envs/vipe

# A local checkout of VIPe (for trajectory metrics).
#   git clone https://github.com/Snap-Research/vipe.git $VIPE_ROOT
export VIPE_ROOT=/path/to/vipe

# A local checkout of VBench (for image quality & motion smoothness).
#   git clone https://github.com/Vchitect/VBench.git $VBENCH_ROOT
# Optional — if unset, defaults to `$IWORLD_BENCH_ROOT/third_party/VBench`.
export VBENCH_ROOT=/path/to/VBench
```

### Video Quality Metrics (no GPU needed)

Use the conda Python interpreter directly (do not rely on `conda activate`
inside scripts):

```bash
$VIPE_ENV/bin/python3 your_script.py
```

### Trajectory Metrics (GPU required)

Trajectory metrics depend on VIPe and need a few extra environment variables:

```bash
LD_LIBRARY_PATH=$VIPE_ENV/lib:$VIPE_ENV/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
VIPE_ROOT=$VIPE_ROOT \
PYTHONPATH=$IWORLD_BENCH_ROOT:$PYTHONPATH \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
$VIPE_ENV/bin/python3 your_script.py
```

**Why these are needed:**
- `LD_LIBRARY_PATH`: makes CUDA/PyTorch native libs visible to the VIPe CUDA extension (`vipe_ext.so`)
- `VIPE_ROOT`: tells `_vipe_worker.py` where to find the VIPe source tree
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`: keep HuggingFace fully offline (only needed if you have already pre-cached every model locally)
- `PYTHONPATH`: ensures `unified_video_metrics.py` and `_vipe_worker.py` can be found by subprocess workers

### VBench Metrics (GPU required)

Imaging Quality and Motion Smoothness are backed by **VBench**, an external
project we do not ship with this repository. You need to install it yourself.

#### 1. Clone VBench

Either place it at the default location next to this folder:

```bash
# Default: third_party/VBench/ next to unified_video_metrics.py
git clone https://github.com/Vchitect/VBench \
    /path/to/this/repo/third_party/VBench
```

Or put VBench anywhere else and point `VBENCH_ROOT` at it:

```bash
export VBENCH_ROOT=/your/path/to/VBench
```

If neither is set up, calling any VBench metric will raise a `FileNotFoundError`
with instructions.

#### 2. Install VBench dependencies

Follow the official setup (conda env, Python deps, etc.) from:

https://github.com/Vchitect/VBench

#### 3. Download VBench model weights

VBench's internal `model_manager.py` looks for model checkpoints under a
cache directory controlled by `VBENCH_CACHE_DIR` (default: VBench's own
default). For Image Quality and Motion Smoothness you need:

| Model | Expected sub-path under `$VBENCH_CACHE_DIR` |
|---|---|
| MUSIQ (Image Quality) | `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` |
| AMT-S (Motion Smoothness) | `amt_model/amt-s.pth` |

Download them via the official VBench instructions, then export:

```bash
export VBENCH_CACHE_DIR=/your/path/to/vbench_model_cache
```

#### 4. Verify

```bash
python3 unified_video_metrics.py /videos /out --metric vbench --vbench-gpu 0
```

If VBench is missing, you'll get a clear error pointing back to this section.

If your VBench dependencies live outside the Python environment used to run
iWorldBench, prepend that environment's `site-packages`:

```bash
PYTHONPATH=/path/to/vbench_env/lib/python3.10/site-packages:$VBENCH_ROOT:$PYTHONPATH \
$VIPE_ENV/bin/python3 unified_video_metrics.py /videos /out --metric vbench --vbench-gpu 0
```

---

## Metric Reference

### 1. Brightness Consistency

**What it measures**: Temporal stability of luminance across frames.
High score = stable lighting throughout the video.

**Function**: `calculate_brightness_consistency(video_dir, save_dir, max_workers=4)`

**Implementation**: `index_revise_pro_plus_c_h.py` (relative to repo root)
→ `calculate_brightness(video_dir, save_dir, ...)`

**Output CSV**: `reports/video_Brightness_<dataset>_L15.csv`

**Example**:
```python
from unified_video_metrics import calculate_brightness_consistency
calculate_brightness_consistency(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
)
```

---

### 2. Color Temperature Constraints (Hue Consistency)

**What it measures**: Consistency of color hue (warm/cool tone) across frames.
High score = stable white balance / color grading.

**Function**: `calculate_color_temperature(video_dir, save_dir, max_workers=4)`

**Implementation**: `index_revise_pro_plus_c_h.py` (relative to repo root)
→ `calculate_hue(video_dir, save_dir, ...)`

**Output CSV**: `reports/video_Hue_<dataset>_L15.csv`

**Example**:
```python
from unified_video_metrics import calculate_color_temperature
calculate_color_temperature(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
)
```

---

### 3. Video Noise (BRISQUE)

**What it measures**: Perceptual image quality using the BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
Lower BRISQUE score = less noise / higher perceptual quality.

**Function**: `calculate_video_noise(video_dir, save_dir, max_workers=4)`

**Implementation**: `index_revise_pro_plus_c_h.py` (relative to repo root)
→ `calculate_noise(video_dir, save_dir, ...)`

**Output CSV**: `reports/video_Noise_BRISQUE_<dataset>_BRISQUE.csv`

**Example**:
```python
from unified_video_metrics import calculate_video_noise
calculate_video_noise(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
)
```

---

### 4. Sharpness Retention (Tenengrad)

**What it measures**: Frame-level sharpness using the Tenengrad gradient magnitude.
High score = consistently sharp frames, no temporal blur degradation.

**Function**: `calculate_sharpness_retention(video_dir, save_dir, max_workers=4)`

**Implementation**: `index_revise_pro_plus_c_h.py` (relative to repo root)
→ `calculate_clarity(video_dir, save_dir, ...)`

**Output CSV**: `reports/video_Clarity_Tenengrad_<dataset>_K3_T0.5.csv`

**Example**:
```python
from unified_video_metrics import calculate_sharpness_retention
calculate_sharpness_retention(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
)
```

---

### 5. Memory Symmetry

**What it measures**: Pixel-level temporal consistency — whether visually similar frames recur symmetrically (e.g., a forward-backward camera motion returns to the same appearance).
High score = strong visual memory / scene consistency.

**Function**: `calculate_memory_symmetry(video_dir, save_dir, max_workers=4)`

**Implementation**: `index_revise_pro_plus_c_h.py` (relative to repo root)
→ `calculate_memory(video_dir, save_dir, ...)`

**Output CSV**: `reports/video_Memory_MSE_<dataset>_alpha0.1.csv`

**Example**:
```python
from unified_video_metrics import calculate_memory_symmetry
calculate_memory_symmetry(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
)
```

---

## Legacy Video-Only Diagnostic Bundle

```python
import sys
sys.path.insert(0, "/path/to/iworld-bench")  # repo root
from unified_video_metrics import calculate_all_video_quality

calculate_all_video_quality(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    max_workers=4,
)
```

Runs Brightness, Color Temperature, Noise (BRISQUE), Sharpness, and Memory Symmetry in sequence.
No GPU required. Results appear in `{save_dir}/reports/`. This compatibility bundle is useful for diagnostics, but the paper's **Generation Quality** group is `Image Quality + Brightness Consistency + Color Temperature Constraint + Sharpness Retention`.

---

### 6. Trajectory Accuracy

**What it measures**: How closely the generated camera trajectory matches the ground-truth camera extrinsics.
High score = generated video faithfully follows the intended camera path.

**Prerequisite**: Pose NPZ files are generated automatically via VIPe (GPU required).

**Function**: `calculate_trajectory_accuracy(video_dir, save_dir, max_workers=4, camera_txt_dir=None)`

**Implementation**: `index_att2.py` (relative to repo root)
→ `calculate_trajectory_accuracy(video_dir=pose_dir, save_dir, ...)`

**Note**: `video_dir` here is the directory of source `.mp4` videos, NOT the pose directory.
VIPe runs automatically and writes pose NPZ files to `{save_dir}/vipe_output/pose/`.

**Output CSV**: `reports/video_Trajectory_Acc_<dataset>_*.csv`

**Example**:
```python
from unified_video_metrics import calculate_trajectory_accuracy
calculate_trajectory_accuracy(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    camera_txt_dir="./camera_trajectories/inference_txt",
)
```

---

### 7. Trajectory Alignment (Symmetry)

**What it measures**: Forward/backward motion consistency of the generated trajectory.
Evaluates whether a camera that moves forward and then backward returns to its origin.
High score = physically plausible, symmetric camera motion.

**Prerequisite**: Pose NPZ files generated via VIPe (GPU required).

**Function**: `calculate_trajectory_alignment(video_dir, save_dir, max_workers=4)`

**Implementation**: `index_att2.py` (relative to repo root)
→ `calculate_trajectory_difference(video_dir=pose_dir, save_dir, ...)`

**Output CSV**: `reports/video_Trajectory_Diff_<dataset>_*.csv`

**Example**:
```python
from unified_video_metrics import calculate_trajectory_alignment
calculate_trajectory_alignment(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
)
```

---

### 8. Trajectory Tolerance (NPZ Similarity)

**What it measures**: Cosine similarity between the VIPe-estimated trajectory of
the generated video and a target/ground-truth trajectory saved as NPZ.
High score = generated motion closely follows the reference control trajectory.

**Prerequisite**: Pose NPZ files generated via VIPe for generated videos (GPU required).
Also requires a `source_npz_dir` containing target/reference pose NPZ files.

**Function**: `calculate_trajectory_tolerance(video_dir, save_dir, source_npz_dir=None, max_workers=4)`

**Arguments**:
- `source_npz_dir`: directory containing target/reference pose NPZ files.
  Each generated video NPZ is first matched by filename stem, then by parsed
  `camera_<level>_<translation>_<rotation>.npz` or `memory_<id>.npz` in this directory.
  For portable/open-source use, pass this path explicitly.
  **If no matching reference NPZ is found, the video is recorded as "skipped" in the CSV.**

**Implementation**: `index_att2.py` (relative to repo root)
→ `calculate_trajectory_npz_similarity_v2(video_dir=pose_dir, save_dir, source_npz_dir, ...)`

**Output CSV**: `reports/video_TrajectoryTolerance_<dataset>_*.csv`

**Example**:
```python
from unified_video_metrics import calculate_trajectory_tolerance
calculate_trajectory_tolerance(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    source_npz_dir="/path/to/reference/pose/npz/",
)
```

---

## Legacy Trajectory Diagnostic Bundle

Running all trajectory metrics together is more efficient — VIPe runs only once and the
resulting NPZ files are reused by all three metrics. This compatibility bundle contains `Trajectory Accuracy`, `Trajectory Alignment`, and `Trajectory Tolerance`; in the paper taxonomy, `Trajectory Alignment` belongs to **Memory Ability**, while `Motion Smoothness` belongs to **Trajectory Following**.

```python
import sys
sys.path.insert(0, "/path/to/iworld-bench")  # repo root
from unified_video_metrics import calculate_all_trajectory

calculate_all_trajectory(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    source_npz_dir="/path/to/reference/pose/npz/",
    max_workers=4,
)
```

---

### 9. Imaging Quality (VBench / MUSIQ)

**What it measures**: Frame-level perceptual image quality using the VBench MUSIQ-based implementation.
High score = better visual image quality.

**Function**: `calculate_imaging_quality(video_dir, save_dir, gpu="0", overwrite=False, retry_failed=True, limit=None)`

**Implementation**: `$VBENCH_ROOT/scripts/compute_imaging_quality_v3.py` (external dependency)
→ `process_model(model_name, model_dir, args, ...)`

**Output CSV**: `reports/video_Imaging_Quality_MUSIQ_<dataset>_vbench.csv`

**Example**:
```python
from unified_video_metrics import calculate_imaging_quality
calculate_imaging_quality(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    gpu="0",
)
```

---

### 10. Motion Smoothness (VBench / AMT)

**What it measures**: Motion interpolation smoothness using the VBench AMT-based implementation.
High score = smoother temporal motion.

**Function**: `calculate_motion_smoothness(video_dir, save_dir, gpu="0", overwrite=False, retry_failed=True, limit=None)`

**Implementation**: `$VBENCH_ROOT/scripts/compute_motion_smoothness_v3.py` (external dependency)
→ `process_model(model_name, model_dir, args, ...)`

**Output CSV**: `reports/video_Motion_Smoothness_AMT_<dataset>_vbench.csv`

**Example**:
```python
from unified_video_metrics import calculate_motion_smoothness
calculate_motion_smoothness(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    gpu="0",
)
```

---

## Both VBench Metrics at Once

```python
import sys
sys.path.insert(0, "/path/to/iworld-bench")  # repo root
from unified_video_metrics import calculate_all_vbench

calculate_all_vbench(
    video_dir="/path/to/generated_videos",
    save_dir="/path/to/results",
    gpu="0",
)
```

VBench progress/cache files are written under `{save_dir}/vbench_metrics/`.
Unified CSV reports are written under `{save_dir}/reports/`.

---

## Command-Line Interface

All commands assume `$IWORLD_BENCH_ROOT`, `$VIPE_ENV`, `$VIPE_ROOT`, `$VBENCH_ROOT`
are already exported (see [Required Paths](#required-paths)).

For one-click runs with preflight checks, prefer:

```bash
$VIPE_ENV/bin/python3 $IWORLD_BENCH_ROOT/run_iworldbench_evaluation.py \
  /path/to/videos /path/to/results \
  --metric all \
  --source-npz-dir $IWORLD_BENCH_ROOT/camera_trajectories/reference_npz \
  --camera-txt-dir $IWORLD_BENCH_ROOT/camera_trajectories/inference_txt \
  --iworld-root $IWORLD_BENCH_ROOT \
  --vipe-root $VIPE_ROOT \
  --vbench-root $VBENCH_ROOT \
  --vbench-gpu 0
```

```bash
UVM=$IWORLD_BENCH_ROOT/unified_video_metrics.py

# Video-quality metrics only (no GPU needed)
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric brightness
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric color_temperature
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric noise
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric sharpness
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric memory
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric video_quality

# Trajectory metrics (GPU required)
LD_LIBRARY_PATH=$VIPE_ENV/lib:$VIPE_ENV/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
VIPE_ROOT=$VIPE_ROOT \
PYTHONPATH=$IWORLD_BENCH_ROOT:$PYTHONPATH \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric trajectory --source-npz-dir $IWORLD_BENCH_ROOT/camera_trajectories/reference_npz --camera-txt-dir $IWORLD_BENCH_ROOT/camera_trajectories/inference_txt

# VBench metrics (GPU required)
PYTHONPATH=/path/to/vbench_env/site-packages:$VBENCH_ROOT:$IWORLD_BENCH_ROOT:$PYTHONPATH \
VBENCH_ROOT=$VBENCH_ROOT \
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric vbench --vbench-gpu 0
PYTHONPATH=/path/to/vbench_env/site-packages:$VBENCH_ROOT:$IWORLD_BENCH_ROOT:$PYTHONPATH \
VBENCH_ROOT=$VBENCH_ROOT \
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric imaging_quality --vbench-gpu 0
PYTHONPATH=/path/to/vbench_env/site-packages:$VBENCH_ROOT:$IWORLD_BENCH_ROOT:$PYTHONPATH \
VBENCH_ROOT=$VBENCH_ROOT \
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric motion_smoothness --vbench-gpu 0

# All 9 paper metrics (GPU required for trajectory + VBench)
LD_LIBRARY_PATH=$VIPE_ENV/lib:$VIPE_ENV/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
VIPE_ROOT=$VIPE_ROOT \
PYTHONPATH=/path/to/vbench_env/site-packages:$VBENCH_ROOT:$IWORLD_BENCH_ROOT:$PYTHONPATH \
VBENCH_ROOT=$VBENCH_ROOT \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
$VIPE_ENV/bin/python3 $UVM /path/to/videos /path/to/results --metric all --source-npz-dir $IWORLD_BENCH_ROOT/camera_trajectories/reference_npz --camera-txt-dir $IWORLD_BENCH_ROOT/camera_trajectories/inference_txt --vbench-gpu 0

# Task-specific paper metrics
$VIPE_ENV/bin/python3 $UVM /path/to/diff_videos /path/to/action_results --metric action_control --source-npz-dir $IWORLD_BENCH_ROOT/camera_trajectories/reference_npz --camera-txt-dir $IWORLD_BENCH_ROOT/camera_trajectories/inference_txt --vbench-gpu 0
$VIPE_ENV/bin/python3 $UVM /path/to/mem_videos /path/to/memory_results --metric memory_ability
$VIPE_ENV/bin/python3 $UVM /path/to/camera_following_videos /path/to/camera_following_results --metric camera_following --source-npz-dir $IWORLD_BENCH_ROOT/camera_trajectories/source_reference_npz --vbench-gpu 0
```

Available `--metric` values:

| Value | Description | GPU |
|---|---|---|
| `brightness` | Brightness Consistency | No |
| `color_temperature` | Color Temperature / Hue Consistency | No |
| `noise` | Noise (BRISQUE) | No |
| `sharpness` | Sharpness Retention (Tenengrad) | No |
| `memory` | Memory Symmetry | No |
| `video_quality` | All 5 video-quality metrics above | No |
| `traj_accuracy` | Trajectory Accuracy | Yes |
| `traj_alignment` | Trajectory Alignment (Symmetry) | Yes |
| `traj_tolerance` | Trajectory Tolerance (NPZ Similarity) | Yes |
| `trajectory` | All 3 trajectory metrics (single VIPe pass) | Yes |
| `imaging_quality` | VBench Imaging Quality (MUSIQ) | Yes |
| `motion_smoothness` | VBench Motion Smoothness (AMT) | Yes |
| `vbench` | Both VBench metrics | Yes |
| `generation_quality` | Paper group: Image Quality, Brightness Consistency, Color Temperature Constraint, Sharpness Retention | Image Quality requires VBench |
| `trajectory_following` | Paper group: Motion Smoothness, Trajectory Accuracy, Trajectory Tolerance | Yes |
| `memory_ability` | Paper group: Memory Symmetry, Trajectory Alignment | Trajectory Alignment requires VIPe |
| `action_control` | Recommended paper metrics for `Diff`/action tasks: Generation Quality + Trajectory Following | Yes |
| `camera_following` | Optional paper metrics for trajectory-input models: Generation Quality + Motion Smoothness + Trajectory Tolerance against original camera trajectories | Yes |
| `all` | All 9 paper metrics | Yes |

---

## Notes

### Video-quality metrics (no GPU)

- Work directly on `.mp4` files; no GPU or pose estimation required.
- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.ts`, `.flv`, `.webm`

### Trajectory metrics (GPU required)

- Require **VIPe** (pose estimation pipeline) to extract camera trajectories from raw video.
- VIPe runs automatically when any trajectory metric is called. It distributes videos across
  GPUs 0, 5, 6, 7 using one subprocess per GPU.
- **NPZ caching**: once a video is processed, its pose NPZ file is saved to
  `{save_dir}/vipe_output/pose/{video_stem}.npz`. On re-runs, videos with existing NPZ files
  are **skipped automatically** — no redundant computation.
- VIPe requires the environment variables listed in the **Environment Setup** section above.
  Missing `LD_LIBRARY_PATH` will cause the CUDA extension (`vipe_ext.so`) to fail to load.

### Trajectory Tolerance — `source_npz_dir`

- Generated videos do **not** contain a reliable ground-truth camera trajectory. VIPe estimates
  the generated video's trajectory first, then the metric compares that estimated trajectory
  against a target/GT/reference trajectory stored as NPZ.
- The reference pose NPZ can either have the **same filename stem** as the generated video NPZ
  or use the parsed control name such as `camera_<level>_<translation>_<rotation>.npz`
  or `memory_<id>.npz`.
  Example: generated video `scene01_zoom_camera_1_0_5.mp4` → reference NPZ `camera_1_0_5.npz`.
- If no matching reference NPZ exists, that video is recorded as **"skipped"** in the output CSV.
  This is expected behavior when running on test data without reference trajectories.
- `source_npz_dir` should be passed explicitly for reproducible runs.
- This package includes default resources:
  - inference TXT: `camera_trajectories/inference_txt/`
  - evaluation NPZ: `camera_trajectories/reference_npz/`
  - original-camera NPZ for trajectory-input models: `camera_trajectories/source_reference_npz/`

### Trajectory Accuracy vs Trajectory Tolerance

- **Trajectory Accuracy** compares the VIPe-estimated trajectory of the generated video against
  the command trajectory TXT, e.g. `camera_<level>_<translation>_<rotation>.txt` or `memory_<id>.txt`.
- **Trajectory Tolerance** compares the VIPe-estimated trajectory of the generated video against
  a target/GT/reference trajectory stored as NPZ. In datasets where the ideal trajectory starts
  from the packaged control TXT files, those TXT trajectories should be converted or exported
  to the matching NPZ representation before running this metric.
- **CameraFollowing** uses `Trajectory Tolerance` with `source_reference_npz/`, not `Trajectory Accuracy`, because the target is the original source-video camera trajectory rather than a discrete `camera_<level>_<translation>_<rotation>.txt` command.

### VBench metrics

- `calculate_imaging_quality` wraps VBench MUSIQ-based imaging quality.
- `calculate_motion_smoothness` wraps VBench AMT-based motion smoothness.
- VBench raw progress files are stored under `{save_dir}/vbench_metrics/<metric_name>/`.
- Unified detail/summary CSV files are exported under `{save_dir}/reports/`.
- Use `--vbench-gpu` or the `gpu` argument to select the GPU.

### Required Model Weights

These metrics depend on the following pretrained models. Download them once
and point the listed environment variables / cache locations at them.

**VIPe (trajectory metrics)** — standard HuggingFace / torch.hub caches:

| Model | Default cache (overridable via standard HF / torch envs) |
|---|---|
| GeoCalib (pinhole.tar) | `~/.cache/torch/hub/geocalib/` |
| bert-base-uncased | `~/.cache/huggingface/hub/models--bert-base-uncased/` |
| UniDepthV2 vitl14 | `~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/` |
| Video-Depth-Anything-Small | `~/.cache/torch/hub/checkpoints/video_depth_anything_vits.pth` |
| GroundingDINO | `~/.cache/torch/hub/checkpoints/groundingdino_swint_ogc.pth` |
| Prior-Depth-Anything vitb | `~/.cache/huggingface/hub/models--Rain729--Prior-Depth-Anything/` |

**VBench (image quality, motion smoothness)** — controlled by `VBENCH_CACHE_DIR`:

| Model | Sub-path under `$VBENCH_CACHE_DIR` |
|---|---|
| MUSIQ | `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` |
| AMT-S | `amt_model/amt-s.pth` |

When running fully offline (e.g. air-gapped clusters), also export:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
