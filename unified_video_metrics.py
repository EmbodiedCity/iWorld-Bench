#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified video evaluation metrics.

Public API — all functions take (video_dir, save_dir, ...) with video files as input.
VIPe labeling runs automatically for trajectory metrics and stores NPZ output at
{save_dir}/vipe_output/.  Results (CSV) are written to save_dir.

Usage example:
    from unified_video_metrics import calculate_all

    calculate_all(
        video_dir="/path/to/generated_videos",
        save_dir="/path/to/results",
        source_npz_dir="/path/to/reference_npz",   # optional, for trajectory tolerance
    )
"""

import os
import sys
import json
import csv
import importlib
import logging
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List

import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VIPE_ROOT = os.path.join(_THIS_DIR, "third_party", "vipe")
_VIPE_ROOT = os.environ.get("VIPE_ROOT") or _DEFAULT_VIPE_ROOT
_DEFAULT_CAMERA_TXT_DIR = os.path.join(_THIS_DIR, "camera_trajectories", "inference_txt")
_DEFAULT_REFERENCE_NPZ_DIR = os.path.join(_THIS_DIR, "camera_trajectories", "reference_npz")
_DEFAULT_SOURCE_REFERENCE_NPZ_DIR = os.path.join(_THIS_DIR, "camera_trajectories", "source_reference_npz")

# VBench is an external dependency: https://github.com/Vchitect/VBench
# Users should either:
#   1. clone VBench into `third_party/VBench/` next to this file (default), OR
#   2. set the VBENCH_ROOT environment variable to their own VBench checkout.
_DEFAULT_VBENCH_ROOT = os.path.join(_THIS_DIR, "third_party", "VBench")
_VBENCH_ROOT = os.environ.get("VBENCH_ROOT") or _DEFAULT_VBENCH_ROOT
_VBENCH_SCRIPTS_DIR = os.path.join(_VBENCH_ROOT, "scripts")

sys.path.insert(0, _THIS_DIR)
sys.path.append(_VIPE_ROOT)

# ─── Import metric implementation functions ───────────────────────────────────
from index_revise_pro_plus_c_h import (
    calculate_brightness as _brightness,
    calculate_hue as _hue,
    calculate_noise as _noise,
    calculate_clarity as _clarity,
    calculate_memory as _memory,
)
from index_att2 import (
    calculate_trajectory_accuracy as _traj_acc,
    calculate_trajectory_difference as _traj_diff,
    calculate_trajectory_npz_similarity_v2 as _traj_sim,
)

# ─── VIPe configuration ───────────────────────────────────────────────────────
AVAILABLE_GPUS: List[int] = [0, 5, 6, 7]
POSE_DIR_NAME: str = "pose"
SUPPORTED_FORMATS: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.ts', '.flv', '.webm')

# Path to the standalone worker script, located alongside this file.
_WORKER_SCRIPT = os.path.join(_THIS_DIR, "_vipe_worker.py")

_VBENCH_SHORT_NAMES = {
    "imaging_quality": "Imaging_Quality_MUSIQ",
    "motion_smoothness": "Motion_Smoothness_AMT",
}


def _ensure_vbench_path() -> None:
    """Validate that VBench is available locally, then add it to sys.path.

    VBench is an external project. We do not ship its source. Users must
    either clone it into the default location or point VBENCH_ROOT at their
    own checkout before calling any VBench-backed metric.
    """
    if not os.path.isdir(_VBENCH_SCRIPTS_DIR):
        raise FileNotFoundError(
            "VBench scripts not found at: {}\n"
            "\n"
            "VBench is an external dependency. To enable the VBench-backed\n"
            "metrics (image quality, motion smoothness), please either:\n"
            "  1. Clone VBench into {!r} (default location):\n"
            "       git clone https://github.com/Vchitect/VBench {}\n"
            "  2. Or set the VBENCH_ROOT environment variable to your own\n"
            "     VBench checkout, e.g.:\n"
            "       export VBENCH_ROOT=/path/to/your/VBench\n"
            "\n"
            "See README_metrics.md (\"VBench Metrics\" section) for full setup\n"
            "instructions, including model weight downloads."
            .format(_VBENCH_SCRIPTS_DIR, _DEFAULT_VBENCH_ROOT, _DEFAULT_VBENCH_ROOT)
        )
    for path in (_VBENCH_ROOT, _VBENCH_SCRIPTS_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)


def _select_vbench_gpu(gpu: str, torch_module) -> None:
    if not torch_module.cuda.is_available():
        return
    first_gpu = str(gpu).split(",")[0].strip()
    if not first_gpu:
        return
    try:
        gpu_idx = int(first_gpu)
    except ValueError:
        return
    if gpu_idx < torch_module.cuda.device_count():
        torch_module.cuda.set_device(gpu_idx)


def _get_dataset_name(input_dir: str) -> str:
    path_parts = os.path.normpath(input_dir).split(os.sep)
    if len(path_parts) >= 2:
        return f"{path_parts[-2]}_{path_parts[-1]}"
    return Path(input_dir).name


def _write_vbench_report_csv(
    metric_name: str,
    video_dir: str,
    save_dir: str,
    metric_dir: str,
    summary,
) -> None:
    short_name = _VBENCH_SHORT_NAMES[metric_name]
    report_dir = os.path.join(save_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    dataset_name = _get_dataset_name(video_dir)
    video_csv = os.path.join(report_dir, f"video_{short_name}_{dataset_name}_vbench.csv")
    summary_csv = os.path.join(report_dir, f"summary_{short_name}_{dataset_name}_vbench.csv")
    results_file = os.path.join(metric_dir, "results.jsonl")

    scores = {}
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("status") == "done" and item.get("score") is not None:
                    scores[item["video_name"]] = float(item["score"])

    with open(video_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Video_Name", f"{short_name}_Score"])
        for name, score in sorted(scores.items()):
            writer.writerow([name, round(score, 6)])

    avg_score = summary.mean if summary and summary.mean is not None else (
        float(np.mean(list(scores.values()))) if scores else 0.0
    )
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{short_name}_Avg_Score"])
        writer.writerow([round(avg_score, 6)])

    print(f"\n[Done] VBench report saved: {video_csv}")


def _run_vbench_metric(
    metric_name: str,
    video_dir: str,
    save_dir: str,
    gpu: str = "0",
    overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """Internal helper that runs one VBench metric and exports the unified CSV."""
    _ensure_vbench_path()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if metric_name == "imaging_quality":
        module_name = "compute_imaging_quality_v3"
    elif metric_name == "motion_smoothness":
        module_name = "compute_motion_smoothness_v3"
    else:
        raise ValueError(f"Unsupported VBench metric: {metric_name}")
    metric_module = importlib.import_module(module_name)
    model_manager_module = importlib.import_module("model_manager")
    _select_vbench_gpu(gpu, model_manager_module.torch)

    # VBench's process_model requires `skip_first_frame` on its args namespace;
    # we always evaluate every frame, so it is hard-coded to False here.
    args = SimpleNamespace(
        generated_subdir="",
        metrics_subdir=os.path.join(save_dir, "vbench_metrics"),
        overwrite=overwrite,
        retry_failed=retry_failed,
        limit=limit,
        skip_first_frame=False,
    )

    model_name = Path(video_dir).name
    try:
        summary = metric_module.process_model(model_name, video_dir, args)
    finally:
        model_manager_module.MetricModelManager.unload_all()
    metric_dir = os.path.join(save_dir, "vbench_metrics", metric_name)
    _write_vbench_report_csv(metric_name, video_dir, save_dir, metric_dir, summary)
    return summary


def _is_labeled(video_path: str, vipe_output_dir: str) -> bool:
    """Return True if a pose NPZ already exists for this video."""
    stem = Path(video_path).stem
    return (Path(vipe_output_dir) / POSE_DIR_NAME / f"{stem}.npz").exists()


def _run_vipe(video_dir: str, vipe_output_dir: str) -> str:
    """
    Run VIPe on all videos in video_dir using subprocess-per-GPU workers.
    Using subprocess.Popen instead of mp.Process avoids the multiprocessing
    bootstrap error that occurs when VIPe's internal code spawns further
    child processes inside an already-spawned worker.
    Returns the pose NPZ directory path.
    """
    logger = logging.getLogger("vipe_runner")
    if not os.path.isdir(_VIPE_ROOT):
        raise FileNotFoundError(
            f"VIPe checkout not found: {_VIPE_ROOT}. "
            "Set VIPE_ROOT=/path/to/vipe or clone VIPe into third_party/vipe."
        )
    os.makedirs(vipe_output_dir, exist_ok=True)

    video_paths: List[str] = []
    for ext in SUPPORTED_FORMATS:
        video_paths.extend(
            str(p) for p in Path(video_dir).rglob(f"*{ext}")
            if not p.name.startswith('.')
        )

    if not video_paths:
        logger.warning(f"No videos found in {video_dir}")
        return os.path.join(vipe_output_dir, POSE_DIR_NAME)

    already_done = sum(1 for v in video_paths if _is_labeled(v, vipe_output_dir))
    logger.info(
        f"VIPe: {len(video_paths)} videos total, "
        f"{already_done} already labeled, "
        f"{len(video_paths) - already_done} to process"
    )

    if already_done < len(video_paths):
        gpus = AVAILABLE_GPUS
        # Distribute videos round-robin across GPUs
        chunks: List[List[str]] = [[] for _ in gpus]
        for i, v in enumerate(video_paths):
            chunks[i % len(gpus)].append(v)

        tmp_files: List[str] = []
        processes: List[subprocess.Popen] = []

        for idx, rank in enumerate(gpus):
            chunk = chunks[idx]
            if not chunk:
                continue
            # Write the video list to a temp JSON file
            fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=f"vipe_gpu{rank}_")
            os.close(fd)
            with open(tmp_path, "w") as f:
                json.dump(chunk, f)
            tmp_files.append(tmp_path)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            env["VIPE_ROOT"] = _VIPE_ROOT

            p = subprocess.Popen(
                [sys.executable, _WORKER_SCRIPT, tmp_path, vipe_output_dir, str(idx)],
                env=env,
            )
            processes.append(p)
            logger.info(f"Launched VIPe worker for GPU {rank} (PID {p.pid}, {len(chunk)} videos)")

        for p in processes:
            p.wait()

        for f in tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass

        logger.info("VIPe labeling complete")

    pose_dir = os.path.join(vipe_output_dir, POSE_DIR_NAME)
    return pose_dir


# ─── Per-metric public APIs ───────────────────────────────────────────────────

def calculate_brightness_consistency(video_dir: str, save_dir: str, max_workers: int = 4):
    """Brightness consistency. Input: video files."""
    _brightness(video_dir=video_dir, save_dir=save_dir, max_workers=max_workers)


def calculate_color_temperature(video_dir: str, save_dir: str, max_workers: int = 4):
    """Color temperature (hue) consistency. Input: video files."""
    _hue(video_dir=video_dir, save_dir=save_dir, max_workers=max_workers)


def calculate_video_noise(video_dir: str, save_dir: str, max_workers: int = 4):
    """Video noise level (BRISQUE). Input: video files."""
    _noise(video_dir=video_dir, save_dir=save_dir, max_workers=max_workers)


def calculate_sharpness_retention(video_dir: str, save_dir: str, max_workers: int = 4):
    """Sharpness retention (Tenengrad). Input: video files."""
    _clarity(video_dir=video_dir, save_dir=save_dir, max_workers=max_workers)


def calculate_memory_symmetry(video_dir: str, save_dir: str, max_workers: int = 4):
    """Memory symmetry (pixel-level frame consistency). Input: video files."""
    _memory(video_dir=video_dir, save_dir=save_dir, max_workers=max_workers)


# ─── Trajectory metrics (VIPe runs automatically) ────────────────────────────

def calculate_trajectory_accuracy(
    video_dir: str,
    save_dir: str,
    max_workers: int = 4,
    camera_txt_dir: Optional[str] = None,
):
    """
    Trajectory accuracy: compares generated camera trajectory against
    ground-truth extrinsics (camera_x_y_z.txt).
    VIPe labeling runs automatically → {save_dir}/vipe_output/pose/
    """
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_acc(
        video_dir=pose_dir,
        save_dir=save_dir,
        max_workers=max_workers,
        target_camera_dir=camera_txt_dir,
    )


def calculate_trajectory_alignment(video_dir: str, save_dir: str, max_workers: int = 4):
    """
    Trajectory symmetry alignment: measures forward/backward motion consistency.
    VIPe labeling runs automatically → {save_dir}/vipe_output/pose/
    """
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_diff(video_dir=pose_dir, save_dir=save_dir, max_workers=max_workers)


def calculate_trajectory_tolerance(
    video_dir: str,
    save_dir: str,
    source_npz_dir: Optional[str] = None,
    max_workers: int = 4,
):
    """
    Trajectory tolerance: cosine similarity between VIPe-estimated generated
    video trajectories and target/GT/reference trajectories saved as NPZ.
    VIPe labeling runs automatically → {save_dir}/vipe_output/pose/
    source_npz_dir: NPZ directory of target/GT/reference trajectories.
                    Falls back to the default configured in index_att2.py.
    """
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_sim(
        video_dir=pose_dir,
        save_dir=save_dir,
        source_npz_dir=source_npz_dir,
        max_workers=max_workers,
    )


def calculate_imaging_quality(
    video_dir: str,
    save_dir: str,
    gpu: str = "0",
    overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """VBench imaging quality (MUSIQ). Input: video files. Requires GPU."""
    return _run_vbench_metric(
        "imaging_quality",
        video_dir,
        save_dir,
        gpu=gpu,
        overwrite=overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )


def calculate_motion_smoothness(
    video_dir: str,
    save_dir: str,
    gpu: str = "0",
    overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """VBench motion smoothness (AMT). Input: video files. Requires GPU."""
    return _run_vbench_metric(
        "motion_smoothness",
        video_dir,
        save_dir,
        gpu=gpu,
        overwrite=overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )


# ─── Composite interfaces ─────────────────────────────────────────────────────

def calculate_generation_quality(
    video_dir: str,
    save_dir: str,
    max_workers: int = 4,
    vbench_gpu: str = "0",
    vbench_overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """Run the paper's Generation Quality metrics: image quality, brightness, color temperature, sharpness."""
    calculate_imaging_quality(
        video_dir,
        save_dir,
        gpu=vbench_gpu,
        overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    calculate_brightness_consistency(video_dir, save_dir, max_workers)
    calculate_color_temperature(video_dir, save_dir, max_workers)
    calculate_sharpness_retention(video_dir, save_dir, max_workers)


def calculate_trajectory_following(
    video_dir: str,
    save_dir: str,
    source_npz_dir: Optional[str] = None,
    max_workers: int = 4,
    camera_txt_dir: Optional[str] = None,
    vbench_gpu: str = "0",
    vbench_overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """Run the paper's Trajectory Following metrics: motion smoothness, trajectory accuracy, trajectory tolerance."""
    calculate_motion_smoothness(
        video_dir,
        save_dir,
        gpu=vbench_gpu,
        overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_acc(
        video_dir=pose_dir,
        save_dir=save_dir,
        max_workers=max_workers,
        target_camera_dir=camera_txt_dir,
    )
    _traj_sim(
        video_dir=pose_dir,
        save_dir=save_dir,
        source_npz_dir=source_npz_dir,
        max_workers=max_workers,
    )


def calculate_memory_ability(video_dir: str, save_dir: str, max_workers: int = 4):
    """Run the paper's Memory Ability metrics: memory symmetry and trajectory alignment."""
    calculate_memory_symmetry(video_dir, save_dir, max_workers)
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_diff(video_dir=pose_dir, save_dir=save_dir, max_workers=max_workers)


def calculate_action_control(
    video_dir: str,
    save_dir: str,
    source_npz_dir: Optional[str] = None,
    max_workers: int = 4,
    camera_txt_dir: Optional[str] = None,
    vbench_gpu: str = "0",
    vbench_overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """Run the recommended metrics for action/Diff tasks: Generation Quality + Trajectory Following."""
    calculate_generation_quality(
        video_dir,
        save_dir,
        max_workers=max_workers,
        vbench_gpu=vbench_gpu,
        vbench_overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    calculate_trajectory_following(
        video_dir,
        save_dir,
        source_npz_dir=source_npz_dir,
        max_workers=max_workers,
        camera_txt_dir=camera_txt_dir,
        vbench_gpu=vbench_gpu,
        vbench_overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )


def calculate_camera_following(
    video_dir: str,
    save_dir: str,
    source_npz_dir: Optional[str] = None,
    max_workers: int = 4,
    vbench_gpu: str = "0",
    vbench_overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """Run metrics for camera-trajectory-input models: Generation Quality + Motion Smoothness + Trajectory Tolerance."""
    reference_npz_dir = source_npz_dir or _DEFAULT_SOURCE_REFERENCE_NPZ_DIR
    calculate_generation_quality(
        video_dir,
        save_dir,
        max_workers=max_workers,
        vbench_gpu=vbench_gpu,
        vbench_overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    calculate_motion_smoothness(
        video_dir,
        save_dir,
        gpu=vbench_gpu,
        overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_sim(
        video_dir=pose_dir,
        save_dir=save_dir,
        source_npz_dir=reference_npz_dir,
        max_workers=max_workers,
    )


def calculate_all_video_quality(video_dir: str, save_dir: str, max_workers: int = 4):
    """Run the legacy video-only bundle: brightness, color temperature, noise, sharpness, memory symmetry."""
    calculate_brightness_consistency(video_dir, save_dir, max_workers)
    calculate_color_temperature(video_dir, save_dir, max_workers)
    calculate_video_noise(video_dir, save_dir, max_workers)
    calculate_sharpness_retention(video_dir, save_dir, max_workers)
    calculate_memory_symmetry(video_dir, save_dir, max_workers)


def calculate_all_trajectory(
    video_dir: str,
    save_dir: str,
    source_npz_dir: Optional[str] = None,
    max_workers: int = 4,
    camera_txt_dir: Optional[str] = None,
):
    """
    Run all 3 trajectory metrics with a single VIPe pass.
    Metrics: trajectory accuracy, trajectory alignment, trajectory tolerance.
    VIPe output reused across all three metrics.
    """
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_acc(
        video_dir=pose_dir,
        save_dir=save_dir,
        max_workers=max_workers,
        target_camera_dir=camera_txt_dir,
    )
    _traj_diff(video_dir=pose_dir, save_dir=save_dir, max_workers=max_workers)
    _traj_sim(
        video_dir=pose_dir,
        save_dir=save_dir,
        source_npz_dir=source_npz_dir,
        max_workers=max_workers,
    )


def calculate_all_vbench(
    video_dir: str,
    save_dir: str,
    gpu: str = "0",
    overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
):
    """Run both VBench metrics (imaging quality and motion smoothness). Requires GPU."""
    summaries = {}
    summaries["imaging_quality"] = calculate_imaging_quality(
        video_dir,
        save_dir,
        gpu=gpu,
        overwrite=overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    summaries["motion_smoothness"] = calculate_motion_smoothness(
        video_dir,
        save_dir,
        gpu=gpu,
        overwrite=overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    return summaries


def calculate_all(
    video_dir: str,
    save_dir: str,
    source_npz_dir: Optional[str] = None,
    max_workers: int = 4,
    vbench_gpu: str = "0",
    vbench_overwrite: bool = False,
    retry_failed: bool = True,
    limit: Optional[int] = None,
    camera_txt_dir: Optional[str] = None,
):
    """
    Run the paper's 9 metrics with a single VIPe pass.

    Args:
        video_dir:       Directory containing generated .mp4 videos.
        save_dir:        Output directory for CSV reports and VIPe NPZ cache.
        source_npz_dir:  NPZ directory for target/GT/reference trajectories
                         used by trajectory tolerance.
                         Optional — falls back to the default in index_att2.py.
        max_workers:     Thread count for metric computation.
        camera_txt_dir:   Directory containing camera_<level>_<translation>_<rotation>.txt
                          files used by trajectory accuracy.
        vbench_gpu:      GPU ID for VBench imaging/motion metrics.
        vbench_overwrite: Recompute existing VBench results.
        retry_failed:    Retry videos previously marked as failed by VBench.
        limit:           Optional cap on the number of videos passed to VBench.
    """
    calculate_imaging_quality(
        video_dir,
        save_dir,
        gpu=vbench_gpu,
        overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    calculate_brightness_consistency(video_dir, save_dir, max_workers)
    calculate_color_temperature(video_dir, save_dir, max_workers)
    calculate_sharpness_retention(video_dir, save_dir, max_workers)
    calculate_motion_smoothness(
        video_dir,
        save_dir,
        gpu=vbench_gpu,
        overwrite=vbench_overwrite,
        retry_failed=retry_failed,
        limit=limit,
    )
    calculate_memory_symmetry(video_dir, save_dir, max_workers)
    pose_dir = _run_vipe(video_dir, os.path.join(save_dir, "vipe_output"))
    _traj_acc(
        video_dir=pose_dir,
        save_dir=save_dir,
        max_workers=max_workers,
        target_camera_dir=camera_txt_dir,
    )
    _traj_sim(
        video_dir=pose_dir,
        save_dir=save_dir,
        source_npz_dir=source_npz_dir,
        max_workers=max_workers,
    )
    _traj_diff(video_dir=pose_dir, save_dir=save_dir, max_workers=max_workers)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compute all iWorldBench video metrics from raw generated videos."
    )
    parser.add_argument("video_dir", help="Directory of generated .mp4 videos")
    parser.add_argument("save_dir", help="Output directory for results")
    parser.add_argument(
        "--source-npz-dir",
        default=None,
        help="NPZ directory for target/GT/reference trajectories (trajectory tolerance metric). "
             "Defaults to camera_trajectories/source_reference_npz for camera_following, "
             "and camera_trajectories/reference_npz otherwise.",
    )
    parser.add_argument(
        "--camera-txt-dir",
        default=_DEFAULT_CAMERA_TXT_DIR,
        help="Directory containing camera_<level>_<translation>_<rotation>.txt files "
             "for trajectory accuracy. Defaults to the packaged camera_trajectories/inference_txt directory.",
    )
    parser.add_argument(
        "--metric",
        default="all",
        choices=[
            "all",
            "brightness", "color_temperature", "noise", "sharpness", "memory",
            "traj_accuracy", "traj_alignment", "traj_tolerance",
            "imaging_quality", "motion_smoothness",
            "generation_quality", "trajectory_following", "memory_ability", "action_control", "camera_following",
            "video_quality", "trajectory", "vbench",
        ],
        help="Which metric(s) to run (default: all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread count for metric computation (default: 4)",
    )
    parser.add_argument(
        "--vbench-gpu",
        default="0",
        help="GPU ID for VBench imaging/motion metrics (default: 0)",
    )
    parser.add_argument(
        "--vbench-overwrite",
        action="store_true",
        help="Recompute existing VBench imaging/motion results",
    )
    parser.add_argument(
        "--skip-failed",
        dest="retry_failed",
        action="store_false",
        default=True,
        help="Do not retry failed VBench imaging/motion videos",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of videos for VBench imaging/motion metrics",
    )
    args = parser.parse_args()
    source_npz_dir = args.source_npz_dir or (
        _DEFAULT_SOURCE_REFERENCE_NPZ_DIR if args.metric == "camera_following" else _DEFAULT_REFERENCE_NPZ_DIR
    )

    _DISPATCH = {
        "all": lambda: calculate_all(
            args.video_dir,
            args.save_dir,
            source_npz_dir,
            args.max_workers,
            args.vbench_gpu,
            args.vbench_overwrite,
            args.retry_failed,
            args.limit,
            args.camera_txt_dir,
        ),
        "generation_quality": lambda: calculate_generation_quality(args.video_dir, args.save_dir, args.max_workers, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
        "trajectory_following": lambda: calculate_trajectory_following(args.video_dir, args.save_dir, source_npz_dir, args.max_workers, args.camera_txt_dir, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
        "memory_ability": lambda: calculate_memory_ability(args.video_dir, args.save_dir, args.max_workers),
        "action_control": lambda: calculate_action_control(args.video_dir, args.save_dir, source_npz_dir, args.max_workers, args.camera_txt_dir, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
        "camera_following": lambda: calculate_camera_following(args.video_dir, args.save_dir, source_npz_dir, args.max_workers, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
        "video_quality": lambda: calculate_all_video_quality(args.video_dir, args.save_dir, args.max_workers),
        "trajectory": lambda: calculate_all_trajectory(args.video_dir, args.save_dir, source_npz_dir, args.max_workers, args.camera_txt_dir),
        "vbench": lambda: calculate_all_vbench(args.video_dir, args.save_dir, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
        "brightness": lambda: calculate_brightness_consistency(args.video_dir, args.save_dir, args.max_workers),
        "color_temperature": lambda: calculate_color_temperature(args.video_dir, args.save_dir, args.max_workers),
        "noise": lambda: calculate_video_noise(args.video_dir, args.save_dir, args.max_workers),
        "sharpness": lambda: calculate_sharpness_retention(args.video_dir, args.save_dir, args.max_workers),
        "memory": lambda: calculate_memory_symmetry(args.video_dir, args.save_dir, args.max_workers),
        "traj_accuracy": lambda: calculate_trajectory_accuracy(args.video_dir, args.save_dir, args.max_workers, args.camera_txt_dir),
        "traj_alignment": lambda: calculate_trajectory_alignment(args.video_dir, args.save_dir, args.max_workers),
        "traj_tolerance": lambda: calculate_trajectory_tolerance(args.video_dir, args.save_dir, source_npz_dir, args.max_workers),
        "imaging_quality": lambda: calculate_imaging_quality(args.video_dir, args.save_dir, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
        "motion_smoothness": lambda: calculate_motion_smoothness(args.video_dir, args.save_dir, args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit),
    }

    _DISPATCH[args.metric]()
