#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VIDEOXFUN_PROJECT_URL = "https://github.com/aigc-apps/VideoX-Fun"
REQUIRED_CONTROL_COLUMNS = ("level", "translation", "rotation")
INPUT_PATH_COLUMNS = ("first_frame_path", "filename")
SOURCE_VIDEO_COLUMNS = ("source_video_path", "source_video_filename", "matched_video_path", "matched_video_filename")
CONTROL_PATH_COLUMNS = ("control_txt_path", "camera_path")
DEFAULT_CAMERA_TXT_DIR = Path(__file__).resolve().parent / "camera_trajectories" / "inference_txt"


def _existing_dir(path: str, name: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{name} not found or not a directory: {p}")
    return p


def _existing_dirs(paths: str, name: str) -> List[Path]:
    resolved = []
    for raw_path in paths.split(os.pathsep):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        resolved.append(_existing_dir(raw_path, name))
    if not resolved:
        raise FileNotFoundError(f"{name} did not contain any directories: {paths}")
    return resolved


def _existing_file(path: str, name: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{name} not found or not a file: {p}")
    return p


def _validate_csv_columns(csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    missing = [col for col in REQUIRED_CONTROL_COLUMNS if col not in columns]
    has_input_path = any(col in columns for col in INPUT_PATH_COLUMNS)
    has_control_path = any(col in columns for col in CONTROL_PATH_COLUMNS)
    if missing:
        if has_control_path:
            missing = []
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected either {list(CONTROL_PATH_COLUMNS)} or control columns: {list(REQUIRED_CONTROL_COLUMNS)}"
        )
    if not has_input_path:
        raise ValueError(
            f"CSV must contain one input path column from: {list(INPUT_PATH_COLUMNS)}"
        )


def _load_videoxfun_worker(videoxfun_root: Path):
    eval_system = videoxfun_root / "evaluation_system"
    if not eval_system.is_dir():
        raise FileNotFoundError(
            f"Cannot find VideoX-Fun evaluation_system at {eval_system}. "
            f"Clone VideoX-Fun from {VIDEOXFUN_PROJECT_URL} and pass --videoxfun-root."
        )
    sys.path.insert(0, str(eval_system))
    from workers.videox_worker import VideoXWorker
    return VideoXWorker


def _resolve_input_path(row: Dict[str, str], assets_root: Path) -> Tuple[Path, str]:
    for column in INPUT_PATH_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            path = Path(value).expanduser()
            resolved = path.resolve() if path.is_absolute() else (assets_root / path).resolve()
            return resolved, column
    raise ValueError(f"Missing input path column. Expected one of: {list(INPUT_PATH_COLUMNS)}")


def _resolve_source_video_path(row: Dict[str, str], source_videos_dirs: Optional[List[Path]]) -> Tuple[Optional[Path], str]:
    for column in SOURCE_VIDEO_COLUMNS:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve(), column
        if source_videos_dirs is None:
            return None, column
        candidates = []
        for source_videos_dir in source_videos_dirs:
            candidates.extend([(source_videos_dir / path).resolve(), (source_videos_dir / path.name).resolve()])
        for candidate in candidates:
            if candidate.is_file():
                return candidate, column
        return candidates[0], column
    return None, ""


def _resolve_control_path(row: Dict[str, str], cameras_dir: Path) -> Tuple[Optional[Path], str]:
    repo_root = Path(__file__).resolve().parent
    for column in CONTROL_PATH_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            path = Path(value).expanduser()
            if path.is_absolute():
                return path.resolve(), column
            candidates = [(cameras_dir / path).resolve(), (repo_root / path).resolve()]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate, column
            return candidates[0], column
    return None, ""


def _optional_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


def _parse_control_path(control_path: Path, row: Dict[str, str]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], str]:
    stem = control_path.stem
    camera_match = re.fullmatch(r"camera_(\d+)_(\d+)_(\d+)", stem)
    if camera_match:
        level, translation, rotation = (int(camera_match.group(i)) for i in range(1, 4))
        return level, translation, rotation, None, "camera"
    memory_match = re.fullmatch(r"memory_(\d+)", stem)
    if memory_match:
        memory_id = int(memory_match.group(1))
        level = _optional_int(row.get("level", ""))
        translation = _optional_int(row.get("translation", ""))
        rotation = _optional_int(row.get("rotation", ""))
        return level, translation, rotation, memory_id, "memory"
    level = _optional_int(row.get("level", ""))
    translation = _optional_int(row.get("translation", ""))
    rotation = _optional_int(row.get("rotation", ""))
    memory_id = _optional_int(row.get("memory_id", ""))
    return level, translation, rotation, memory_id, "custom"


def _legacy_camera_path(row: Dict[str, str], cameras_dir: Path) -> Tuple[Path, int, int, int, int]:
    level = int((row.get("level") or "").strip())
    translation = int((row.get("translation") or "").strip())
    rotation = int((row.get("rotation") or "").strip())
    camera_filename = f"camera_{level}_{translation}_{rotation}.txt"
    camera_path = cameras_dir / camera_filename
    corrected = 0
    if not camera_path.is_file():
        fallback_matches = sorted(cameras_dir.glob(f"camera_*_{translation}_{rotation}.txt"))
        if len(fallback_matches) == 1:
            camera_path = fallback_matches[0]
            try:
                level = int(camera_path.stem.split("_")[1])
            except (IndexError, ValueError):
                pass
            corrected = 1
    return camera_path, level, translation, rotation, corrected


def _build_samples(csv_path: Path, assets_root: Path, cameras_dir: Path, source_videos_dirs: Optional[List[Path]], levels: Optional[List[int]], tasks: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    level_set = set(levels) if levels is not None else None
    task_set = {task.lower() for task in tasks} if tasks is not None else None
    stats: Dict[str, Any] = {
        "total_in_csv": 0,
        "filtered_by_task": 0,
        "filtered_by_level": 0,
        "missing_input": 0,
        "missing_source_video": 0,
        "missing_control": 0,
        "corrected_camera_level": 0,
        "invalid_rows": 0,
        "valid": 0,
        "missing_input_files": [],
        "missing_source_video_files": [],
        "missing_control_files": [],
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_in_csv"] += 1
            task = (row.get("task") or row.get("task_type") or "").strip()
            if task_set is not None and task.lower() not in task_set:
                stats["filtered_by_task"] += 1
                continue

            control_path, _ = _resolve_control_path(row, cameras_dir)
            corrected = 0
            if control_path is None:
                try:
                    control_path, level, translation, rotation, corrected = _legacy_camera_path(row, cameras_dir)
                    memory_id = None
                    control_type = "camera"
                except ValueError:
                    stats["invalid_rows"] += 1
                    continue
            else:
                try:
                    level, translation, rotation, memory_id, control_type = _parse_control_path(control_path, row)
                except ValueError:
                    stats["invalid_rows"] += 1
                    continue

            if level_set is not None:
                if level is None or level not in level_set:
                    stats["filtered_by_level"] += 1
                    continue

            try:
                input_path, input_column = _resolve_input_path(row, assets_root)
                source_video_path, source_video_column = _resolve_source_video_path(row, source_videos_dirs)
            except ValueError:
                stats["invalid_rows"] += 1
                continue

            if not input_path.is_file():
                stats["missing_input"] += 1
                if len(stats["missing_input_files"]) < 5:
                    stats["missing_input_files"].append(str(input_path))
                continue

            worker_video_path = input_path
            if source_video_column:
                if source_video_path is None or not source_video_path.is_file():
                    stats["missing_source_video"] += 1
                    if len(stats["missing_source_video_files"]) < 5:
                        stats["missing_source_video_files"].append(str(source_video_path or row.get(source_video_column, "")))
                    continue
                worker_video_path = source_video_path

            if not control_path.is_file():
                stats["missing_control"] += 1
                if len(stats["missing_control_files"]) < 5:
                    stats["missing_control_files"].append(str(control_path))
                continue
            stats["corrected_camera_level"] += corrected

            base_id = (row.get("sample_id") or "").strip() or Path(row.get(input_column, "")).stem
            sample_id = f"{base_id}_{control_path.stem}"
            samples.append({
                "sample_id": sample_id,
                "video_path": str(worker_video_path),
                "first_frame_path": str(input_path),
                "source_video_path": str(source_video_path) if source_video_path is not None else "",
                "source_camera_txt_path": (row.get("source_camera_txt_path") or row.get("source_camera_path") or "").strip(),
                "camera_path": str(control_path),
                "control_txt_path": str(control_path),
                "control_type": control_type,
                "prompt": (row.get("prompt") or "A video with smooth camera motion.").strip(),
                "level": level,
                "translation": translation,
                "rotation": rotation,
                "memory_id": memory_id,
                "task": task,
            })
            stats["valid"] += 1

    return samples, stats


def _filter_existing(samples: List[Dict[str, Any]], output_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    generated_dir = output_dir / "generated_videos"
    pending = []
    skipped = 0
    for sample in samples:
        out_file = generated_dir / f"{sample['sample_id']}.mp4"
        if out_file.exists() and out_file.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(sample)
    return pending, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VideoX-Fun camera-control inference from iWorldBench metadata or CSV."
    )
    parser.add_argument("--csv", required=True, help="CSV with first_frame_path or filename, plus control_txt_path or level, translation, rotation")
    parser.add_argument("--assets-root", default=None, help="Root directory used to resolve relative first_frame_path or filename values")
    parser.add_argument("--source-videos-dir", default=None, help=f"Root directory used to resolve relative source_video_path/source_video_filename values; use {os.pathsep!r} to provide multiple roots")
    parser.add_argument("--videos-dir", default=None, help="Backward-compatible alias for --source-videos-dir")
    parser.add_argument("--cameras-dir", default=str(DEFAULT_CAMERA_TXT_DIR), help="Directory containing packaged camera_*.txt and memory_*.txt control files")
    parser.add_argument("--output-dir", required=True, help="Output directory; generated videos go to output-dir/generated_videos")
    parser.add_argument("--videoxfun-root", default=os.environ.get("VIDEOXFUN_ROOT"), help="Local VideoX-Fun checkout root")
    parser.add_argument("--conda-path", default=os.environ.get("VIDEOXFUN_CONDA_PATH", sys.prefix), help="Conda/env prefix used to run VideoX-Fun")
    parser.add_argument("--model-path", default=os.environ.get("VIDEOXFUN_MODEL_PATH"), help="VideoX-Fun model/checkpoint directory")
    parser.add_argument("--transformer-path", default=None, help="Optional transformer checkpoint override")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Optional camera-level filter, e.g. --levels 1 2 3 4")
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task filter, e.g. --tasks Diff Mem")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit after CSV/path validation")
    parser.add_argument("--video-length", type=int, default=81)
    parser.add_argument("--sample-size", type=int, nargs=2, default=[480, 832], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--sampler-name", default="Flow")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=720000)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output videos already exist")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the generated config without running inference")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.videoxfun_root:
        raise ValueError(f"--videoxfun-root is required. Official project: {VIDEOXFUN_PROJECT_URL}")
    if not args.model_path:
        raise ValueError("--model-path is required, or set VIDEOXFUN_MODEL_PATH")

    csv_path = _existing_file(args.csv, "CSV")
    assets_root_arg = args.assets_root or str(csv_path.parent)
    assets_root = _existing_dir(assets_root_arg, "assets-root")
    source_videos_dir_arg = args.source_videos_dir or args.videos_dir
    source_videos_dirs = _existing_dirs(source_videos_dir_arg, "source-videos-dir") if source_videos_dir_arg else None
    cameras_dir = _existing_dir(args.cameras_dir, "cameras-dir")
    videoxfun_root = _existing_dir(args.videoxfun_root, "videoxfun-root")
    conda_path = _existing_dir(args.conda_path, "conda-path")
    model_path = _existing_dir(args.model_path, "model-path")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    py_candidates = [conda_path / "bin" / name for name in ("python3.10", "python3.11", "python3", "python")]
    if not any(p.exists() and os.access(p, os.X_OK) for p in py_candidates):
        raise FileNotFoundError(f"No executable python found under {conda_path / 'bin'}")

    _validate_csv_columns(csv_path)
    VideoXWorker = _load_videoxfun_worker(videoxfun_root)

    samples, stats = _build_samples(csv_path, assets_root, cameras_dir, source_videos_dirs, levels=args.levels, tasks=args.tasks)
    selected_samples = samples[: args.limit] if args.limit is not None else samples
    pending_samples, skipped_count = (selected_samples, 0) if args.overwrite else _filter_existing(selected_samples, output_dir)

    summary = {
        "videoxfun_project": VIDEOXFUN_PROJECT_URL,
        "csv": str(csv_path),
        "assets_root": str(assets_root),
        "source_videos_dir": [str(path) for path in source_videos_dirs] if source_videos_dirs is not None else [],
        "cameras_dir": str(cameras_dir),
        "output_dir": str(output_dir),
        "build_stats": stats,
        "selected_samples": len(selected_samples),
        "skipped_existing": skipped_count,
        "pending_samples": len(pending_samples),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not selected_samples:
        raise RuntimeError("No valid samples after CSV/path validation")
    if args.dry_run:
        return 0
    if not pending_samples:
        result = {
            "success": True,
            "total_samples": len(selected_samples),
            "success_count": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "skipped_count": skipped_count,
            "videos_dir": str(output_dir / "generated_videos"),
            "message": "All selected samples already exist; use --overwrite to regenerate.",
        }
        with (output_dir / "final_results.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    worker = VideoXWorker(
        conda_path=str(conda_path),
        project_path=str(videoxfun_root),
        gpu_ids=[args.gpu],
    )
    config = {
        "model_path": str(model_path),
        "transformer_path": args.transformer_path,
        "samples": pending_samples,
        "levels": args.levels,
        "output_dir": str(output_dir),
        "video_length": args.video_length,
        "sample_size": args.sample_size,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "seed": args.seed,
        "sampler_name": args.sampler_name,
        "fps": args.fps,
        "timeout": args.timeout,
    }
    result = worker.run(config)
    result_clean = {k: v for k, v in result.items() if k not in ("stdout", "stderr")}
    result_clean["skipped_count"] = skipped_count
    result_clean["total_samples_in_selection"] = len(selected_samples)
    result_clean["videoxfun_project"] = VIDEOXFUN_PROJECT_URL
    with (output_dir / "final_results.json").open("w", encoding="utf-8") as f:
        json.dump(result_clean, f, ensure_ascii=False, indent=2)
    print(json.dumps(result_clean, ensure_ascii=False, indent=2))
    return 0 if result_clean.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
