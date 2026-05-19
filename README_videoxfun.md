# VideoX-Fun Reference Demo for iWorldBench

This document describes the `run_videoxfun_inference.py` reference demo. It is not a required part of the benchmark. The script is provided to show how a camera-controllable video generation model can be connected to the iWorldBench evaluation pipeline.

If you are evaluating your own model, use this demo as a template:

1. Read iWorldBench metadata with first-frame paths and explicit control TXT paths.
2. Load the corresponding control trajectory file.
3. Generate one video for each row.
4. Save generated videos under a directory.
5. Run `run_iworldbench_evaluation.py` on that directory.

## External Project

VideoX-Fun is an external project and is not vendored in this repository.

- Official repository: https://github.com/aigc-apps/VideoX-Fun
- Local checkout placeholder: `/path/to/VideoX-Fun`

The wrapper calls a local VideoX-Fun checkout through `--videoxfun-root`.

## Input CSV Format

`run_videoxfun_inference.py` can consume `dataset/all_pack/metadata.csv` directly. The required columns are:

```csv
sample_id,task,first_frame_path,control_txt_path
sample_000,Diff,assets/example/frame_000.jpg,camera_1_0_5.txt
sample_001,Mem,assets/example/frame_001.jpg,memory_1.txt
```

- `first_frame_path`: first-frame image path relative to `--assets-root`
- `control_txt_path`: control TXT file name or path, resolved against `--cameras-dir`
- `task`: optional task label such as `Diff` or `Mem`
- `source_video_filename` or `source_video_path`: optional source video input for wrappers that extract their own start frame
- `source_camera_txt_path`: optional original source-video camera TXT path when it is available in `camera_trajectories/source_camera_txt/`
- `source_camera_npz_path`: optional original source-video camera NPZ reference for `CameraFollowing` evaluation

The packaged `metadata.csv` contains 3,100 rows: 2,800 `Diff` rows and 300 `Mem` rows.
The packaged `camera_following_metadata.csv` contains 401 original-trajectory samples for models that accept camera trajectory input.

When source videos are split across multiple roots, pass them as one `--source-videos-dir` value joined by the system path separator, for example `dir1:dir2` on Linux.

For every row, the control file must exist under `--cameras-dir` or be provided as an absolute path:

```text
--cameras-dir/<control_txt_path>
```

By default, this repository uses `camera_trajectories/inference_txt/`, which contains 81 `camera_*.txt` files and 8 `memory_*.txt` files.

The expected generated output filename is:

```text
<sample_id>_<control_txt_stem>.mp4
```

This naming convention lets trajectory metrics match the generated video to the corresponding packaged TXT/NPZ control file.

## Running the Reference Demo

Smoke-test example:

```bash
PY=/path/to/videox-env/bin/python
ROOT=/path/to/demo_run

$PY /path/to/iWorldBench/run_videoxfun_inference.py \
  --videoxfun-root /path/to/VideoX-Fun \
  --conda-path /path/to/videox-env \
  --model-path /path/to/model \
  --csv /path/to/iWorldBench/dataset/all_pack/metadata.csv \
  --assets-root /path/to/iWorldBench/dataset/all_pack \
  --source-videos-dir /path/to/source_videos \
  --cameras-dir /path/to/iWorldBench/camera_trajectories/inference_txt \
  --output-dir $ROOT/output/videoxfun_demo \
  --gpu 0 \
  --video-length 81 \
  --inference-steps 50
```

For a full-quality run, use production settings such as `--video-length 81 --inference-steps 50`.

For long runs, redirect logs explicitly:

```bash
nohup $PY -u /path/to/iWorldBench/run_videoxfun_inference.py \
  --videoxfun-root /path/to/VideoX-Fun \
  --conda-path /path/to/videox-env \
  --model-path /path/to/model \
  --csv /path/to/iWorldBench/dataset/all_pack/metadata.csv \
  --assets-root /path/to/iWorldBench/dataset/all_pack \
  --source-videos-dir /path/to/source_videos \
  --cameras-dir /path/to/iWorldBench/camera_trajectories/inference_txt \
  --output-dir /path/to/output/videoxfun_demo \
  --gpu 0 \
  > /path/to/output/videoxfun_inference.log 2>&1 &
```

Inference outputs:

```text
<output-dir>/generated_videos/*.mp4
<output-dir>/final_results.json
```

## Evaluating Generated Videos

Use `run_iworldbench_evaluation.py` as the preferred evaluation wrapper. It performs preflight checks before importing or running heavier metric code.

Video quality only:

```bash
python3 /path/to/iWorldBench/run_iworldbench_evaluation.py \
  /path/to/generated_videos /path/to/eval_output \
  --metric video_quality
```

Trajectory metrics:

```bash
export VIPE_ROOT=/path/to/vipe

python3 /path/to/iWorldBench/run_iworldbench_evaluation.py \
  /path/to/generated_videos /path/to/eval_output \
  --metric trajectory \
  --source-npz-dir /path/to/iWorldBench/camera_trajectories/reference_npz \
  --camera-txt-dir /path/to/iWorldBench/camera_trajectories/inference_txt \
  --vipe-root $VIPE_ROOT
```

Camera-following metrics for trajectory-input models:

```bash
export VIPE_ROOT=/path/to/vipe
export VBENCH_ROOT=/path/to/VBench

python3 /path/to/iWorldBench/run_iworldbench_evaluation.py \
  /path/to/camera_following_generated_videos /path/to/camera_following_eval_output \
  --metric camera_following \
  --source-npz-dir /path/to/iWorldBench/camera_trajectories/source_reference_npz \
  --vipe-root $VIPE_ROOT \
  --vbench-root $VBENCH_ROOT \
  --vbench-gpu 0
```

All metrics:

```bash
export VIPE_ROOT=/path/to/vipe
export VBENCH_ROOT=/path/to/VBench

python3 /path/to/iWorldBench/run_iworldbench_evaluation.py \
  /path/to/generated_videos /path/to/eval_output \
  --metric all \
  --source-npz-dir /path/to/iWorldBench/camera_trajectories/reference_npz \
  --camera-txt-dir /path/to/iWorldBench/camera_trajectories/inference_txt \
  --vipe-root $VIPE_ROOT \
  --vbench-root $VBENCH_ROOT \
  --vbench-gpu 0
```

CSV reports are written under:

```text
<eval_output>/reports/
```

## Reference NPZ Files

A reference NPZ is the target camera trajectory saved in the same pose representation used by VIPe outputs.

It is used by `trajectory_tolerance`:

- VIPe first estimates each generated video's camera trajectory and writes generated NPZ files to `<eval_output>/vipe_output/pose/`.
- `trajectory_tolerance` loads the generated-video NPZ and the target/reference NPZ from `--source-npz-dir`.
- The metric compares the two trajectories by frame-wise 6DoF cosine similarity.

The packaged reference NPZ directory is:

```text
camera_trajectories/reference_npz/
```

The packaged original-camera NPZ directory for `CameraFollowing` is:

```text
camera_trajectories/source_reference_npz/
```

The packaged inference TXT directory is:

```text
camera_trajectories/inference_txt/
```

If you do not use VideoX-Fun, implement an equivalent runner for your model that follows the same input/output contract.
