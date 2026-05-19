#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "dataset" / "all_pack" / "metadata.csv"
DEFAULT_CAMERA_FOLLOWING_CSV = REPO_ROOT / "dataset" / "all_pack" / "camera_following_metadata.csv"
DEFAULT_ASSETS_ROOT = REPO_ROOT / "dataset" / "all_pack"
DEFAULT_CAMERAS_DIR = REPO_ROOT / "camera_trajectories" / "inference_txt"
DEFAULT_SOURCE_CAMERAS_DIR = REPO_ROOT / "camera_trajectories" / "source_camera_txt"
MATRIXGAME_CAM_VALUE = 0.1
MATRIXGAME_TRANSLATION_TO_KEYBOARD = {
    0: [0, 0, 0, 0],
    1: [1, 0, 0, 0],
    2: [0, 1, 0, 0],
    3: [0, 0, 1, 0],
    4: [0, 0, 0, 1],
    5: [1, 0, 1, 0],
    6: [1, 0, 0, 1],
    7: [0, 1, 1, 0],
    8: [0, 1, 0, 1],
}
MATRIXGAME_ROTATION_TO_MOUSE = {
    0: [0.0, 0.0],
    1: [MATRIXGAME_CAM_VALUE, 0.0],
    2: [-MATRIXGAME_CAM_VALUE, 0.0],
    3: [0.0, MATRIXGAME_CAM_VALUE],
    4: [0.0, -MATRIXGAME_CAM_VALUE],
    5: [MATRIXGAME_CAM_VALUE, MATRIXGAME_CAM_VALUE],
    6: [MATRIXGAME_CAM_VALUE, -MATRIXGAME_CAM_VALUE],
    7: [-MATRIXGAME_CAM_VALUE, MATRIXGAME_CAM_VALUE],
    8: [-MATRIXGAME_CAM_VALUE, -MATRIXGAME_CAM_VALUE],
}
MATRIXGAME_MEMORY_ACTIONS = {
    "forward": {"keyboard": [1, 0, 0, 0], "mouse": [0.0, 0.0]},
    "backward": {"keyboard": [0, 1, 0, 0], "mouse": [0.0, 0.0]},
    "left": {"keyboard": [0, 0, 1, 0], "mouse": [0.0, 0.0]},
    "right": {"keyboard": [0, 0, 0, 1], "mouse": [0.0, 0.0]},
    "tilt_up": {"keyboard": [0, 0, 0, 0], "mouse": [MATRIXGAME_CAM_VALUE, 0.0]},
    "tilt_down": {"keyboard": [0, 0, 0, 0], "mouse": [-MATRIXGAME_CAM_VALUE, 0.0]},
    "pan_left": {"keyboard": [0, 0, 0, 0], "mouse": [0.0, -MATRIXGAME_CAM_VALUE]},
    "pan_right": {"keyboard": [0, 0, 0, 0], "mouse": [0.0, MATRIXGAME_CAM_VALUE]},
}
MATRIXGAME_MEMORY_ID_TO_ACTIONS = {
    1: ("forward", "backward"),
    2: ("backward", "forward"),
    3: ("left", "right"),
    4: ("right", "left"),
    5: ("tilt_up", "tilt_down"),
    6: ("tilt_down", "tilt_up"),
    7: ("pan_left", "pan_right"),
    8: ("pan_right", "pan_left"),
    9: ("tilt_up", "tilt_down"),
    10: ("tilt_down", "tilt_up"),
}
TEXT_TRANSLATION_TO_SENTENCE = {
    0: "The camera's movement direction remains stationary.",
    1: "The camera pushes forward (W).",
    2: "The camera pulls back (S).",
    3: "The camera moves to the left (A).",
    4: "The camera moves to the right (D).",
    5: "The camera pushes forward and moves to the left (W+A).",
    6: "The camera pushes forward and moves to the right (W+D).",
    7: "The camera pulls back and moves to the left (S+A).",
    8: "The camera pulls back and moves to the right (S+D).",
}
TEXT_ROTATION_TO_SENTENCE = {
    0: "The rotation direction of the camera remains stationary.",
    1: "The camera tilts up.",
    2: "The camera tilts down.",
    3: "The camera pans to the right.",
    4: "The camera pans to the left.",
    5: "The camera tilts up and pans to the right.",
    6: "The camera tilts up and pans to the left.",
    7: "The camera tilts down and pans to the right.",
    8: "The camera tilts down and pans to the left.",
}
TEXT_MEMORY_ACTION_SENTENCES = {
    "forward": "The camera pushes forward (W).",
    "backward": "The camera pulls back (S).",
    "left": "The camera moves to the left (A).",
    "right": "The camera moves to the right (D).",
    "tilt_up": "The camera tilts up.",
    "tilt_down": "The camera tilts down.",
    "pan_left": "The camera pans to the left.",
    "pan_right": "The camera pans to the right.",
}
DEFAULT_TEXT_BASE_PROMPT = "A video with smooth camera motion."

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "cami2v": {
        "display_name": "CamI2V",
        "repo_url": "https://github.com/ZGCTroy/CamI2V",
        "project_env": "CAMI2V_ROOT",
        "python_env": "CAMI2V_PYTHON",
        "path_args": [("--checkpoint-path", "checkpoint_path", "CAMI2V_CHECKPOINT")],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "entrypoint": "scripts/inference.py",
        "params": {
            "model_type": "cami2v",
            "resolution": "512x320",
            "num_iterations": 5,
            "frames_per_iteration": 16,
            "ddim_steps": 25,
            "cfg_scale": 7.5,
            "camera_cfg": 1.0,
            "frame_stride": 8,
            "seed": 123,
        },
    },
    "motionctrl": {
        "display_name": "MotionCtrl",
        "repo_url": "https://github.com/TencentARC/MotionCtrl",
        "project_env": "MOTIONCTRL_ROOT",
        "python_env": "MOTIONCTRL_PYTHON",
        "path_args": [("--checkpoint-path", "checkpoint_path", "MOTIONCTRL_CHECKPOINT")],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "entrypoint": "scripts/inference.py",
        "params": {
            "model_type": "motionctrl",
            "resolution": "256x256",
            "num_iterations": 5,
            "frames_per_iteration": 16,
            "ddim_steps": 25,
            "cfg_scale": 7.5,
            "camera_cfg": 1.0,
            "seed": 123,
        },
    },
    "cameractrl": {
        "display_name": "CameraCtrl",
        "repo_url": "https://github.com/hehao13/CameraCtrl",
        "project_env": "CAMERACTRL_ROOT",
        "python_env": "CAMERACTRL_PYTHON",
        "path_args": [("--checkpoint-path", "checkpoint_path", "CAMERACTRL_CHECKPOINT")],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "entrypoint": "scripts/inference.py",
        "params": {
            "model_type": "cameractrl",
            "resolution": "256x256",
            "num_iterations": 5,
            "frames_per_iteration": 16,
            "ddim_steps": 25,
            "cfg_scale": 7.5,
            "camera_cfg": 1.0,
            "seed": 123,
        },
    },
    "realcami2v": {
        "display_name": "RealCam-I2V",
        "repo_url": "https://github.com/ZGCTroy/RealCam-I2V",
        "project_env": "REALCAMI2V_ROOT",
        "python_env": "REALCAMI2V_PYTHON",
        "path_args": [("--model-path", "model_path", "REALCAMI2V_MODEL_PATH")],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "entrypoint": "scripts/inference.py",
        "params": {
            "model_name": "cogvideox1.5_controlnetxs_realcam-i2v",
            "video_length": 81,
            "sample_size": [512, 896],
            "fps": 16,
            "seed": 43,
            "inference_steps": 25,
            "guidance_scale": 6.0,
            "trace_extract_ratio": 1.0,
            "trace_scale_factor": 1.0,
        },
    },
    "videox": {
        "display_name": "VideoX-Fun Wan 1.3B",
        "repo_url": "https://github.com/aigc-apps/VideoX-Fun",
        "project_env": "VIDEOXFUN_ROOT",
        "python_env": "VIDEOXFUN_PYTHON",
        "path_args": [
            ("--model-path", "model_path", "VIDEOXFUN_MODEL_PATH"),
            ("--transformer-path", "transformer_path", "VIDEOXFUN_TRANSFORMER_PATH"),
        ],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "entrypoint": "scripts/inference.py",
        "params": {
            "video_length": 81,
            "sample_size": [480, 832],
            "inference_steps": 50,
            "guidance_scale": 6.0,
            "shift": 3.0,
            "seed": 43,
            "sampler_name": "Flow",
            "fps": 16,
        },
    },
    "ac3d": {
        "display_name": "AC3D",
        "repo_url": "https://github.com/snap-research/ac3d",
        "project_env": "AC3D_ROOT",
        "python_env": "AC3D_PYTHON",
        "path_args": [
            ("--base-model-path", "base_model_path", "AC3D_BASE_MODEL_PATH"),
            ("--controlnet-model-path", "controlnet_model_path", "AC3D_CONTROLNET_MODEL_PATH"),
        ],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "entrypoint": "scripts/inference.py",
        "params": {
            "video_length": 49,
            "frames_per_iteration": 49,
            "sample_size": [480, 720],
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "controlnet_weights": 1.0,
            "controlnet_guidance_start": 0.0,
            "controlnet_guidance_end": 0.4,
            "seed": 42,
            "fps": 8,
        },
    },
    "matrixgame": {
        "display_name": "MatrixGame-2",
        "repo_url": "https://github.com/SkyworkAI/Matrix-Game",
        "project_env": "MATRIXGAME_ADAPTER_ROOT",
        "python_env": "MATRIXGAME_PYTHON",
        "path_args": [
            ("--config-path", "config_path", "MATRIXGAME_CONFIG_PATH"),
            ("--checkpoint-path", "checkpoint_path", "MATRIXGAME_CHECKPOINT"),
            ("--pretrained-model-path", "pretrained_model_path", "MATRIXGAME_PRETRAINED_MODEL_PATH"),
        ],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "control_mode": "keyboard_mouse",
        "action_entrypoint": "run_matrixgame_action_inference.py",
        "memory_entrypoint": "run_matrixgame_memory_inference.py",
        "params": {
            "num_frames": 81,
            "memory_segments": [1, 40, 40],
            "seed": 1,
            "height": 352,
            "width": 640,
            "keyboard_mapping": "translation_id_to_keyboard_one_hot",
            "mouse_mapping": "rotation_id_to_mouse_delta",
        },
    },
    "wan": {
        "display_name": "Wan2.2 TI2V",
        "repo_url": "https://github.com/Wan-Video/Wan2.2",
        "project_env": "WAN_ADAPTER_ROOT",
        "python_env": "WAN_PYTHON",
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "control_mode": "text_prompt",
        "entrypoint": "run_wan_csv_inference.py",
        "text_command": "wan_csv",
        "params": {
            "prompt_csv_columns": ["Image Filename", "Prompt"],
            "video_length": 121,
            "sample_size": "704,1280",
            "seed": 43,
        },
    },
    "yume": {
        "display_name": "YUME",
        "repo_url": "https://github.com/stdstu12/YUME",
        "project_env": "YUME_ADAPTER_ROOT",
        "python_env": "YUME_PYTHON",
        "path_args": [
            ("--workdir", "workdir", "YUME_ROOT"),
            ("--videos-dir", "videos_dir", "YUME_VIDEO_DIR"),
        ],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "control_mode": "text_prompt",
        "action_entrypoint": "run_yume_csv_inference.py",
        "memory_entrypoint": "run_yume_memory.py",
        "text_command": "yume_wrapper",
        "params": {
            "sample_script": "fastvideo/sample/sample_5b.py",
            "mixed_precision": "bf16",
            "seed": 43,
            "nproc": 1,
        },
    },
    "cogvideox": {
        "display_name": "CogVideoX-5B-I2V",
        "repo_url": "https://github.com/THUDM/CogVideo",
        "project_env": "COGVIDEOX_ADAPTER_ROOT",
        "python_env": "COGVIDEOX_PYTHON",
        "path_args": [("--model_path", "model_path", "COGVIDEOX_MODEL_PATH")],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "control_mode": "text_prompt",
        "entrypoint": "run_cogvideox_csv_inference.py",
        "text_command": "cogvideox_csv",
        "params": {
            "prompt_csv_columns": ["Image Filename", "Prompt"],
            "num_frames": 49,
            "height": 480,
            "width": 720,
            "fps": 10,
        },
    },
    "hunyuan": {
        "display_name": "HunyuanVideo-1.5",
        "repo_url": "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5",
        "project_env": "HUNYUAN_ADAPTER_ROOT",
        "python_env": "HUNYUAN_PYTHON",
        "path_args": [("--model_path", "model_path", "HUNYUAN_MODEL_PATH")],
        "default_tasks": None,
        "supported_tasks": {"diff", "mem"},
        "control_mode": "text_prompt",
        "entrypoint": "run_hunyuan_csv_inference_inprocess_v2.py",
        "text_command": "hunyuan_csv",
        "params": {
            "prompt_csv_columns": ["Image Filename", "Prompt"],
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "seed": 123,
            "enable_step_distill": True,
            "rewrite": False,
        },
    },
    "hyworldplay": {
        "display_name": "HY-WorldPlay",
        "repo_url": "https://github.com/Tencent-Hunyuan/HY-WorldPlay",
        "project_env": "HYWORLDPLAY_ROOT",
        "python_env": "HYWORLDPLAY_PYTHON",
        "path_args": [
            ("--model_path", "model_path", "HYWORLDPLAY_MODEL_PATH"),
            ("--action_ckpt", "action_ckpt", "HYWORLDPLAY_ACTION_CKPT"),
        ],
        "default_csv": DEFAULT_CAMERA_FOLLOWING_CSV,
        "default_cameras_dir": DEFAULT_SOURCE_CAMERAS_DIR,
        "default_tasks": ["CameraFollowing"],
        "supported_tasks": {"camerafollowing", "camera_following"},
        "control_mode": "source_camera_trajectory",
        "entrypoint": "run_camera_trajectory_inference.py",
        "source_videos_env": "HYWORLDPLAY_SOURCE_VIDEOS_DIR",
        "params": {
            "num_frames": 77,
            "width": 832,
            "height": 480,
            "num_inference_steps": 4,
            "seed": 1,
            "total_gpus": 1,
        },
    },
}


def optional_int(value: Any) -> Optional[int]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    return int(float(text))


def normalize_tasks(tasks: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not tasks:
        return None
    return [task.strip() for task in tasks if task and task.strip()]


def canonical_task_key(task: str) -> str:
    return task.strip().lower().replace("_", "")


def resolve_candidate_path(value: str, primary_base: Path, repo_root: Path) -> Path:
    raw = Path(value.strip())
    if raw.is_absolute():
        return raw
    candidates = [primary_base / raw, repo_root / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_control_path(row: Dict[str, str], cameras_dir: Path) -> Tuple[Path, str]:
    explicit = (row.get("control_txt_path") or row.get("camera_path") or "").strip()
    if explicit:
        raw = Path(explicit)
        if raw.is_absolute():
            return raw, "control_txt_path"
        candidates = [cameras_dir / raw, cameras_dir / raw.name, REPO_ROOT / raw]
        for candidate in candidates:
            if candidate.exists():
                return candidate, "control_txt_path"
        return candidates[0], "control_txt_path"
    level = optional_int(row.get("level") or row.get("\u7ea7\u522b"))
    translation = optional_int(row.get("translation") or row.get("\u5e73\u52a8"))
    rotation = optional_int(row.get("rotation") or row.get("\u8f6c\u52a8"))
    if level is None or translation is None or rotation is None:
        raise ValueError("missing control_txt_path or legacy level/translation/rotation columns")
    return cameras_dir / f"camera_{level}_{translation}_{rotation}.txt", "legacy_camera_columns"


def resolve_source_camera_path(row: Dict[str, str], cameras_dir: Path) -> Tuple[Path, str]:
    explicit = (row.get("source_camera_txt_path") or row.get("source_camera_path") or row.get("camera_path") or "").strip()
    if not explicit:
        raise ValueError("missing source_camera_txt_path")
    raw = Path(explicit)
    if raw.is_absolute():
        return raw, "source_camera_txt_path"
    candidates = [REPO_ROOT / raw, cameras_dir / raw, cameras_dir / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate, "source_camera_txt_path"
    return candidates[0], "source_camera_txt_path"


def parse_control_metadata(control_path: Path, row: Dict[str, str]) -> Dict[str, Any]:
    stem = control_path.stem
    if stem.startswith("camera_"):
        parts = stem.split("_")
        if len(parts) != 4:
            raise ValueError(f"invalid camera control name: {control_path.name}")
        return {
            "control_type": "camera",
            "level": optional_int(row.get("level")) if row.get("level") else int(parts[1]),
            "translation": optional_int(row.get("translation")) if row.get("translation") else int(parts[2]),
            "rotation": optional_int(row.get("rotation")) if row.get("rotation") else int(parts[3]),
            "memory_id": None,
        }
    if stem.startswith("memory_"):
        parts = stem.split("_")
        memory_id = optional_int(row.get("memory_id")) if row.get("memory_id") else int(parts[1])
        return {
            "control_type": "memory",
            "level": optional_int(row.get("level")) if row.get("level") else None,
            "translation": optional_int(row.get("translation")) if row.get("translation") else None,
            "rotation": optional_int(row.get("rotation")) if row.get("rotation") else None,
            "memory_id": memory_id,
        }
    raise ValueError(f"unsupported control file name: {control_path.name}")


def source_pair_key(filename: str, camera: bool = False) -> str:
    stem = Path(filename).stem
    return stem.replace("_camera_", "_") if camera else stem.replace("_video_", "_")


def source_camera_control_metadata(row: Dict[str, str], cameras_dir: Path) -> Tuple[Path, str, Dict[str, Any]]:
    camera_path, control_column = resolve_source_camera_path(row, cameras_dir)
    source_video_filename = (row.get("source_video_filename") or row.get("source_video_path") or row.get("matched_video_filename") or row.get("matched_video_path") or "").strip()
    camera_key = source_pair_key(camera_path.name, camera=True)
    video_key = source_pair_key(source_video_filename, camera=False) if source_video_filename else ""
    return camera_path, control_column, {
        "control_type": "source_camera",
        "control_stem": camera_path.stem,
        "level": None,
        "translation": None,
        "rotation": None,
        "memory_id": None,
        "source_video_filename": source_video_filename,
        "source_camera_txt_path": str(camera_path),
        "source_pair_key": camera_key,
        "source_video_pair_key": video_key,
        "source_pair_key_matches": bool(video_key and video_key == camera_key),
    }


def matrixgame_metadata_control_stem(row: Dict[str, str]) -> Optional[str]:
    explicit = (row.get("control_txt_path") or row.get("camera_path") or "").strip()
    return Path(explicit).stem if explicit else None


def matrixgame_control_metadata(row: Dict[str, str], task: str) -> Dict[str, Any]:
    stem = matrixgame_metadata_control_stem(row)
    level = optional_int(row.get("level") or row.get("\u7ea7\u522b"))
    translation = optional_int(row.get("translation") or row.get("\u5e73\u52a8"))
    rotation = optional_int(row.get("rotation") or row.get("\u8f6c\u52a8"))
    memory_id = optional_int(row.get("memory_id") or row.get("\u5e8f\u53f7"))
    if stem and stem.startswith("camera_"):
        parts = stem.split("_")
        if len(parts) == 4:
            level = level if level is not None else int(parts[1])
            translation = translation if translation is not None else int(parts[2])
            rotation = rotation if rotation is not None else int(parts[3])
    if stem and stem.startswith("memory_"):
        parts = stem.split("_")
        if len(parts) == 2:
            memory_id = memory_id if memory_id is not None else int(parts[1])
    if task.lower() == "diff":
        if translation not in MATRIXGAME_TRANSLATION_TO_KEYBOARD or rotation not in MATRIXGAME_ROTATION_TO_MOUSE:
            raise ValueError("MatrixGame Diff samples require translation and rotation ids in [0, 8]")
        control_stem = stem if stem and stem.startswith("camera_") else f"camera_{level or 0}_{translation}_{rotation}"
        keyboard = MATRIXGAME_TRANSLATION_TO_KEYBOARD[translation]
        mouse = MATRIXGAME_ROTATION_TO_MOUSE[rotation]
        return {
            "control_type": "keyboard_mouse_action",
            "control_column": "keyboard_mouse_from_translation_rotation",
            "control_stem": control_stem,
            "level": level,
            "translation": translation,
            "rotation": rotation,
            "memory_id": None,
            "matrixgame_control": {
                "mode": "action_single_segment",
                "num_frames": 81,
                "keyboard_cond_shape": [81, 4],
                "mouse_cond_shape": [81, 2],
                "keyboard": keyboard,
                "mouse": mouse,
                "repeat_frames": 81,
            },
        }
    if task.lower() == "mem":
        if memory_id not in MATRIXGAME_MEMORY_ID_TO_ACTIONS:
            raise ValueError("MatrixGame Mem samples require memory_id in [1, 10]")
        first_action, second_action = MATRIXGAME_MEMORY_ID_TO_ACTIONS[memory_id]
        first = MATRIXGAME_MEMORY_ACTIONS[first_action]
        second = MATRIXGAME_MEMORY_ACTIONS[second_action]
        control_stem = stem if stem and stem.startswith("memory_") else f"memory_{memory_id}"
        return {
            "control_type": "keyboard_mouse_memory",
            "control_column": "keyboard_mouse_from_memory_id",
            "control_stem": control_stem,
            "level": None,
            "translation": None,
            "rotation": None,
            "memory_id": memory_id,
            "matrixgame_control": {
                "mode": "memory_two_segment",
                "num_frames": 81,
                "keyboard_cond_shape": [81, 4],
                "mouse_cond_shape": [81, 2],
                "segments": [
                    {"name": "static", "start": 0, "end": 0, "keyboard": [0, 0, 0, 0], "mouse": [0.0, 0.0]},
                    {"name": first_action, "start": 1, "end": 40, **first},
                    {"name": second_action, "start": 41, "end": 80, **second},
                ],
            },
        }
    raise ValueError(f"unsupported MatrixGame task: {task}")


def text_prompt_control_metadata(row: Dict[str, str], task: str) -> Dict[str, Any]:
    stem = matrixgame_metadata_control_stem(row)
    level = optional_int(row.get("level") or row.get("\u7ea7\u522b"))
    translation = optional_int(row.get("translation") or row.get("\u5e73\u52a8"))
    rotation = optional_int(row.get("rotation") or row.get("\u8f6c\u52a8"))
    memory_id = optional_int(row.get("memory_id") or row.get("\u5e8f\u53f7"))
    base_prompt = (row.get("prompt") or DEFAULT_TEXT_BASE_PROMPT).strip()
    if stem and stem.startswith("camera_"):
        parts = stem.split("_")
        if len(parts) == 4:
            level = level if level is not None else int(parts[1])
            translation = translation if translation is not None else int(parts[2])
            rotation = rotation if rotation is not None else int(parts[3])
    if stem and stem.startswith("memory_"):
        parts = stem.split("_")
        if len(parts) == 2:
            memory_id = memory_id if memory_id is not None else int(parts[1])
    if task.lower() == "diff":
        if translation not in TEXT_TRANSLATION_TO_SENTENCE or rotation not in TEXT_ROTATION_TO_SENTENCE:
            raise ValueError("Text-control Diff samples require translation and rotation ids in [0, 8]")
        movement_prompt = TEXT_TRANSLATION_TO_SENTENCE[translation]
        rotation_prompt = TEXT_ROTATION_TO_SENTENCE[rotation]
        prompt = " ".join([movement_prompt, rotation_prompt, base_prompt]).strip()
        control_stem = stem if stem and stem.startswith("camera_") else f"camera_{level or 0}_{translation}_{rotation}"
        return {
            "control_type": "text_prompt_action",
            "control_column": "text_prompt_from_translation_rotation",
            "control_stem": control_stem,
            "level": level,
            "translation": translation,
            "rotation": rotation,
            "memory_id": None,
            "prompt": prompt,
            "text_control": {
                "mode": "action_prompt",
                "base_prompt": base_prompt,
                "translation_prompt": movement_prompt,
                "rotation_prompt": rotation_prompt,
                "prompt": prompt,
            },
        }
    if task.lower() == "mem":
        if memory_id not in MATRIXGAME_MEMORY_ID_TO_ACTIONS:
            raise ValueError("Text-control Mem samples require memory_id in [1, 10]")
        first_action, second_action = MATRIXGAME_MEMORY_ID_TO_ACTIONS[memory_id]
        first_prompt = TEXT_MEMORY_ACTION_SENTENCES[first_action]
        second_prompt = TEXT_MEMORY_ACTION_SENTENCES[second_action]
        prompt = " ".join([
            first_prompt,
            f"second, {second_prompt}",
            "Keep the motion magnitude consistent across both movements.",
            base_prompt,
        ]).strip()
        control_stem = stem if stem and stem.startswith("memory_") else f"memory_{memory_id}"
        return {
            "control_type": "text_prompt_memory",
            "control_column": "text_prompt_from_memory_id",
            "control_stem": control_stem,
            "level": None,
            "translation": None,
            "rotation": None,
            "memory_id": memory_id,
            "prompt": prompt,
            "text_control": {
                "mode": "memory_prompt",
                "base_prompt": base_prompt,
                "first_action": first_action,
                "second_action": second_action,
                "first_prompt": first_prompt,
                "second_prompt": second_prompt,
                "prompt": prompt,
            },
        }
    raise ValueError(f"unsupported text-control task: {task}")


def parse_shard(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    left, right = value.split("/", 1)
    index, total = int(left), int(right)
    if total <= 0 or index < 0 or index >= total:
        raise ValueError("--shard must use i/N with 0 <= i < N")
    return index, total


def build_samples(args: argparse.Namespace, spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    csv_path = Path(args.csv).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve()
    cameras_dir = Path(args.cameras_dir).expanduser().resolve()
    requested_tasks = normalize_tasks(args.tasks) or spec.get("default_tasks")
    requested_task_set = {canonical_task_key(task) for task in requested_tasks} if requested_tasks else None
    supported_tasks = spec.get("supported_tasks")
    control_mode = spec.get("control_mode", "txt")
    level_set = set(args.levels or []) if args.levels else None
    shard = parse_shard(args.shard)
    samples: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_in_csv": 0,
        "filtered_by_task": 0,
        "filtered_by_model_support": 0,
        "filtered_by_level": 0,
        "filtered_by_shard": 0,
        "missing_input": 0,
        "missing_control": 0,
        "invalid_rows": 0,
        "valid": 0,
        "task_counts": {},
        "control_counts": {},
        "missing_input_examples": [],
        "missing_control_examples": [],
        "source_pair_key_mismatch": 0,
        "source_pair_key_mismatch_examples": [],
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            stats["total_in_csv"] += 1
            task = (row.get("task") or row.get("task_type") or "Diff").strip()
            task_key = canonical_task_key(task)
            if requested_task_set is not None and task_key not in requested_task_set:
                stats["filtered_by_task"] += 1
                continue
            if supported_tasks is not None and task_key not in supported_tasks:
                stats["filtered_by_model_support"] += 1
                continue
            if shard is not None and row_index % shard[1] != shard[0]:
                stats["filtered_by_shard"] += 1
                continue
            try:
                input_value = (row.get("first_frame_path") or row.get("image_path") or row.get("filename") or row.get("\u6587\u4ef6\u540d") or "").strip()
                if not input_value:
                    raise ValueError("missing first_frame_path")
                input_path = resolve_candidate_path(input_value, assets_root, REPO_ROOT)
                if control_mode == "source_camera_trajectory":
                    control_path, control_column, control_meta = source_camera_control_metadata(row, cameras_dir)
                elif control_mode == "keyboard_mouse":
                    control_path = None
                    control_meta = matrixgame_control_metadata(row, task)
                    control_column = control_meta["control_column"]
                elif control_mode == "text_prompt":
                    control_path = None
                    control_meta = text_prompt_control_metadata(row, task)
                    control_column = control_meta["control_column"]
                else:
                    control_path, control_column = resolve_control_path(row, cameras_dir)
                    control_meta = parse_control_metadata(control_path, row)
            except Exception:
                stats["invalid_rows"] += 1
                continue
            if level_set is not None:
                level = control_meta.get("level")
                if level is None or level not in level_set:
                    stats["filtered_by_level"] += 1
                    continue
            if not input_path.is_file():
                stats["missing_input"] += 1
                if len(stats["missing_input_examples"]) < 5:
                    stats["missing_input_examples"].append(str(input_path))
                continue
            if control_path is not None and not control_path.is_file():
                stats["missing_control"] += 1
                if len(stats["missing_control_examples"]) < 5:
                    stats["missing_control_examples"].append(str(control_path))
                continue
            base_id = (row.get("sample_id") or input_path.stem).strip()
            control_stem = control_meta.get("control_stem") or control_path.stem
            sample_id = base_id if base_id == control_stem else f"{base_id}_{control_stem}"
            sample = {
                "sample_id": sample_id,
                "task": task,
                "dataset": (row.get("dataset") or "").strip(),
                "first_frame_path": str(input_path),
                "video_path": str(input_path),
                "control_txt_path": str(control_path) if control_path is not None else "",
                "camera_path": str(control_path) if control_path is not None else "",
                "metadata_control_txt_path": (row.get("control_txt_path") or row.get("camera_path") or "").strip(),
                "control_column": control_column,
                "prompt": (row.get("prompt") or "A video with smooth camera motion.").strip(),
                **control_meta,
            }
            if control_mode == "source_camera_trajectory" and not sample.get("source_pair_key_matches"):
                stats["source_pair_key_mismatch"] += 1
                if len(stats["source_pair_key_mismatch_examples"]) < 5:
                    stats["source_pair_key_mismatch_examples"].append({
                        "source_video_filename": sample.get("source_video_filename"),
                        "source_camera_txt_path": sample.get("source_camera_txt_path"),
                        "source_video_pair_key": sample.get("source_video_pair_key"),
                        "source_pair_key": sample.get("source_pair_key"),
                    })
            samples.append(sample)
            stats["valid"] += 1
            stats["task_counts"][task] = stats["task_counts"].get(task, 0) + 1
            stats["control_counts"][control_meta["control_type"]] = stats["control_counts"].get(control_meta["control_type"], 0) + 1
            if args.max_samples is not None and args.max_samples > 0 and len(samples) >= args.max_samples:
                break
    return samples, stats


def output_video_path(args: argparse.Namespace, sample: Dict[str, Any]) -> str:
    return str(Path(args.output_dir).expanduser() / "generated_videos" / f"{sample['sample_id']}.mp4")


def text_prompt_csv_path(model_key: str, args: argparse.Namespace) -> str:
    return str(Path(args.output_dir).expanduser() / f"{model_key}_text_prompt_manifest.csv")


def yume_adapter_csv_path(args: argparse.Namespace, task: str) -> str:
    suffix = "memory" if task.lower() == "mem" else "action"
    return str(Path(args.output_dir).expanduser() / f"yume_{suffix}_manifest.csv")


def prompt_csv_image_filename(sample: Dict[str, Any], args: argparse.Namespace) -> str:
    first_frame = Path(sample["first_frame_path"])
    assets_root = Path(args.assets_root).expanduser().resolve()
    try:
        return str(first_frame.resolve().relative_to(assets_root))
    except ValueError:
        return first_frame.name


def build_prompt_csv_rows(samples: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, str]]:
    return [
        {"Image Filename": prompt_csv_image_filename(sample, args), "Prompt": sample["prompt"]}
        for sample in samples
    ]


def yume_video_filename(sample: Dict[str, Any]) -> str:
    return Path(sample["first_frame_path"]).with_suffix(".mp4").name


def build_yume_action_csv_rows(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "\u6587\u4ef6\u540d": yume_video_filename(sample),
            "\u7ea7\u522b": sample.get("level"),
            "\u5e73\u52a8": sample.get("translation"),
            "\u8f6c\u52a8": sample.get("rotation"),
        }
        for sample in samples
        if sample["task"].lower() == "diff"
    ]


def build_yume_memory_csv_rows(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "\u6587\u4ef6\u540d": yume_video_filename(sample),
            "\u5e8f\u53f7": sample.get("memory_id"),
        }
        for sample in samples
        if sample["task"].lower() == "mem"
    ]


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def env_or_placeholder(env_name: str) -> str:
    return os.environ.get(env_name) or f"<{env_name}>"


def model_arg_value(args: argparse.Namespace, field: str, env_name: str) -> str:
    return getattr(args, field) or env_or_placeholder(env_name)


def build_config(model_key: str, args: argparse.Namespace, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    config = {
        "model": model_key,
        "display_name": spec["display_name"],
        "original_repo": spec.get("repo_url"),
        "dry_run_only": True,
        "gpu": args.gpu,
        "samples": list(samples),
        "output_dir": str(Path(args.output_dir).expanduser()),
        **spec["params"],
    }
    if spec.get("control_mode") == "text_prompt":
        config["prompt_csv_path"] = text_prompt_csv_path(model_key, args)
        config["prompt_csv_rows"] = build_prompt_csv_rows(samples, args)
        if model_key == "yume":
            config["yume_action_csv_path"] = yume_adapter_csv_path(args, "Diff")
            config["yume_memory_csv_path"] = yume_adapter_csv_path(args, "Mem")
            config["yume_action_csv_rows"] = build_yume_action_csv_rows(samples)
            config["yume_memory_csv_rows"] = build_yume_memory_csv_rows(samples)
    if spec.get("control_mode") == "source_camera_trajectory":
        config["source_videos_dir"] = args.source_videos_dir or env_or_placeholder(spec["source_videos_env"])
        config["source_cameras_dir"] = str(Path(args.cameras_dir).expanduser())
    for _, field, env_name in spec.get("path_args", []):
        config[field] = model_arg_value(args, field, env_name)
    return config


def build_command(model_key: str, args: argparse.Namespace, sample: Dict[str, Any]) -> List[str]:
    spec = MODEL_SPECS[model_key]
    if spec.get("control_mode") == "source_camera_trajectory":
        project_root = Path(args.project_root) if args.project_root else Path(env_or_placeholder(spec["project_env"]))
        source_videos_dir = args.source_videos_dir or env_or_placeholder(spec["source_videos_env"])
        command = [
            "env",
            f"CUDA_VISIBLE_DEVICES={args.gpu}",
            args.python,
            str(project_root / spec["entrypoint"]),
            "--videos_dir",
            source_videos_dir,
            "--cameras_dir",
            str(Path(args.cameras_dir).expanduser()),
            "--output_dir",
            str(Path(args.output_dir).expanduser()),
            "--model_path",
            model_arg_value(args, "model_path", "HYWORLDPLAY_MODEL_PATH"),
            "--action_ckpt",
            model_arg_value(args, "action_ckpt", "HYWORLDPLAY_ACTION_CKPT"),
            "--num_frames",
            str(spec["params"]["num_frames"]),
            "--width",
            str(spec["params"]["width"]),
            "--height",
            str(spec["params"]["height"]),
            "--num_inference_steps",
            str(spec["params"]["num_inference_steps"]),
            "--seed",
            str(spec["params"]["seed"]),
            "--max_samples",
            str(args.max_samples or 0),
            "--gpu_id",
            str(args.gpu),
            "--total_gpus",
            str(args.total_gpus),
        ]
        return command
    if spec.get("control_mode") == "text_prompt":
        project_root = Path(args.project_root) if args.project_root else Path(env_or_placeholder(spec["project_env"]))
        entrypoint_name = spec.get("entrypoint")
        if sample["task"].lower() == "mem" and spec.get("memory_entrypoint"):
            entrypoint_name = spec["memory_entrypoint"]
        elif sample["task"].lower() == "diff" and spec.get("action_entrypoint"):
            entrypoint_name = spec["action_entrypoint"]
        command = [args.python, str(project_root / entrypoint_name)]
        generated_dir = str(Path(args.output_dir).expanduser() / "generated_videos")
        prompt_csv = text_prompt_csv_path(model_key, args)
        if spec["text_command"] == "wan_csv":
            command.extend([
                "--image_base_path", str(Path(args.assets_root).expanduser()),
                "--csv_path", prompt_csv,
                "--save_path", generated_dir,
                "--gpu", str(args.gpu),
                "--video_length", str(spec["params"]["video_length"]),
                "--sample_size", str(spec["params"]["sample_size"]),
                "--seed", str(spec["params"]["seed"]),
            ])
        elif spec["text_command"] == "cogvideox_csv":
            command.extend([
                "--model_path", model_arg_value(args, "model_path", "COGVIDEOX_MODEL_PATH"),
                "--input_csv_path", prompt_csv,
                "--image_dir", str(Path(args.assets_root).expanduser()),
                "--output_dir", generated_dir,
                "--gpu", str(args.gpu),
            ])
        elif spec["text_command"] == "hunyuan_csv":
            command.extend([
                "--csv_file", prompt_csv,
                "--pre_pic_dir", str(Path(args.assets_root).expanduser()),
                "--output_dir", generated_dir,
                "--model_path", model_arg_value(args, "model_path", "HUNYUAN_MODEL_PATH"),
                "--resolution", spec["params"]["resolution"],
                "--aspect_ratio", spec["params"]["aspect_ratio"],
                "--seed", str(spec["params"]["seed"]),
                "--enable_step_distill", str(spec["params"]["enable_step_distill"]).lower(),
                "--rewrite", str(spec["params"]["rewrite"]).lower(),
                "--cuda_visible_devices", str(args.gpu),
            ])
        elif spec["text_command"] == "yume_wrapper":
            command.extend([
                "--csv", yume_adapter_csv_path(args, sample["task"]),
                "--videos-dir", model_arg_value(args, "videos_dir", "YUME_VIDEO_DIR"),
                "--output-dir", str(Path(args.output_dir).expanduser()),
                "--gpus", str(args.gpu),
                "--nproc", str(spec["params"]["nproc"]),
                "--workdir", model_arg_value(args, "workdir", "YUME_ROOT"),
                "--sample-script", spec["params"]["sample_script"],
                "--python", args.python,
                "--base-prompt", DEFAULT_TEXT_BASE_PROMPT,
                "--seed", str(spec["params"]["seed"]),
                "--mixed-precision", spec["params"]["mixed_precision"],
            ])
            if sample["task"].lower() == "diff":
                levels = args.levels or [sample.get("level") if sample.get("level") is not None else 0]
                command.extend(["--levels", *[str(level) for level in levels]])
        return command
    if spec.get("control_mode") == "keyboard_mouse":
        project_root = Path(args.project_root) if args.project_root else Path(env_or_placeholder(spec["project_env"]))
        entrypoint_key = "memory_entrypoint" if sample["task"].lower() == "mem" else "action_entrypoint"
        command = [
            args.python,
            str(project_root / spec[entrypoint_key]),
            "--gpu",
            str(args.gpu),
        ]
        for field, flag in [("config_path", "--config_path"), ("checkpoint_path", "--checkpoint_path"), ("pretrained_model_path", "--pretrained_model_path")]:
            env_name = {
                "config_path": "MATRIXGAME_CONFIG_PATH",
                "checkpoint_path": "MATRIXGAME_CHECKPOINT",
                "pretrained_model_path": "MATRIXGAME_PRETRAINED_MODEL_PATH",
            }[field]
            value = model_arg_value(args, field, env_name)
            if value:
                command.extend([flag, value])
        command.extend([
            "--num_frames",
            str(spec["params"]["num_frames"]),
            "--output_dir",
            str(Path(args.output_dir).expanduser()),
        ])
        if sample["task"].lower() == "mem":
            command.extend([
                "--memory_csv",
                str(Path(args.csv).expanduser()),
                "--action_dict_csv",
                str(REPO_ROOT / "camera_trajectories" / "memory_dic_with_text_description.csv"),
                "--videos_dir",
                str(Path(args.assets_root).expanduser()),
                "--action_ids",
                str(sample["memory_id"]),
                "--debug_conditions",
            ])
        else:
            command.extend([
                "--video_csv",
                str(Path(args.csv).expanduser()),
                "--videos_dir",
                str(Path(args.assets_root).expanduser()),
                "--levels",
                str(sample.get("level") if sample.get("level") is not None else 0),
            ])
        return command
    project_root = Path(args.project_root) if args.project_root else Path(env_or_placeholder(spec["project_env"]))
    entrypoint = project_root / spec["entrypoint"]
    command = [args.python, str(entrypoint), "--input-image", sample["first_frame_path"], "--control-txt", sample["control_txt_path"], "--output", output_video_path(args, sample), "--gpu", str(args.gpu)]
    for flag, field, env_name in spec.get("path_args", []):
        value = model_arg_value(args, field, env_name)
        if value:
            command.extend([flag, value])
    return command


def validate_model_paths(model_key: str, args: argparse.Namespace) -> List[str]:
    spec = MODEL_SPECS[model_key]
    missing = []
    if args.project_root and not Path(args.project_root).expanduser().exists():
        missing.append(f"project_root={args.project_root}")
    for _, field, _ in spec.get("path_args", []):
        value = getattr(args, field)
        if value and not Path(value).expanduser().exists():
            missing.append(f"{field}={value}")
    if spec.get("control_mode") == "source_camera_trajectory" and args.source_videos_dir and not Path(args.source_videos_dir).expanduser().exists():
        missing.append(f"source_videos_dir={args.source_videos_dir}")
    return missing


def add_common_args(parser: argparse.ArgumentParser, model_key: str) -> None:
    spec = MODEL_SPECS[model_key]
    parser.add_argument("--csv", default=str(spec.get("default_csv", DEFAULT_CSV)), help="Metadata CSV with first_frame_path and control/source-camera columns")
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT), help="Root directory used to resolve first_frame_path")
    parser.add_argument("--cameras-dir", default=str(spec.get("default_cameras_dir", DEFAULT_CAMERAS_DIR)), help="Directory containing packaged control TXT files")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "demo_outputs" / model_key), help="Dry-run output root used for command construction")
    parser.add_argument("--tasks", nargs="+", default=None, choices=["Diff", "Mem", "CameraFollowing", "diff", "mem", "camerafollowing", "camera_following"], help="Task modes to validate")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Optional Diff/action difficulty levels to validate")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for fast smoke validation")
    parser.add_argument("--shard", default=None, help="Optional shard spec i/N, for example 0/4")
    parser.add_argument("--gpu", default="0", help="GPU id used only in constructed commands")
    parser.add_argument("--python", default=os.environ.get(spec["python_env"], "python3"), help="Python executable used only in constructed commands")
    parser.add_argument("--project-root", default=os.environ.get(spec["project_env"], ""), help=f"Model project root, or set {spec['project_env']}")
    parser.add_argument("--print-samples", type=int, default=3, help="How many sample command examples to print")
    parser.add_argument("--check-model-paths", action="store_true", help="Also require provided model/project paths to exist")
    parser.add_argument("--write-manifest", default=None, help="Optional JSON manifest path to write dry-run config")
    if spec.get("control_mode") == "source_camera_trajectory":
        parser.add_argument("--source-videos-dir", default=os.environ.get(spec["source_videos_env"], ""), help=f"Source video directory passed to the external runner, or set {spec['source_videos_env']}")
        parser.add_argument("--total-gpus", type=int, default=spec["params"]["total_gpus"], help="Total GPU shards used by the external runner")
    for flag, field, env_name in spec.get("path_args", []):
        parser.add_argument(flag, dest=field, default=os.environ.get(env_name, ""), help=f"Optional model path, or set {env_name}")


def print_report(model_key: str, args: argparse.Namespace, samples: Sequence[Dict[str, Any]], stats: Dict[str, Any]) -> None:
    spec = MODEL_SPECS[model_key]
    print("=" * 80)
    print(f"iWorldBench dry-run demo: {spec['display_name']}")
    print("=" * 80)
    print(f"CSV: {Path(args.csv).expanduser()}")
    print(f"Assets root: {Path(args.assets_root).expanduser()}")
    if spec.get("repo_url"):
        print(f"Original GitHub: {spec['repo_url']}")
    if spec.get("control_mode") == "keyboard_mouse":
        print("Control mode: keyboard/mouse one-hot schedule")
    elif spec.get("control_mode") == "text_prompt":
        print("Control mode: text prompt from action/memory metadata")
        print(f"Prompt CSV preview path: {text_prompt_csv_path(model_key, args)}")
    elif spec.get("control_mode") == "source_camera_trajectory":
        print("Control mode: source video directory + original source-camera TXT directory")
        print(f"Source videos dir: {args.source_videos_dir or env_or_placeholder(spec['source_videos_env'])}")
        print(f"Source cameras dir: {Path(args.cameras_dir).expanduser()}")
    else:
        print(f"Controls dir: {Path(args.cameras_dir).expanduser()}")
    print(f"Output root: {Path(args.output_dir).expanduser()}")
    print(f"Requested tasks: {normalize_tasks(args.tasks) or spec.get('default_tasks') or ['Diff', 'Mem']}")
    print(f"Supported tasks: {sorted(spec['supported_tasks'])}")
    print("Sample validation stats:")
    for key in ["total_in_csv", "filtered_by_task", "filtered_by_model_support", "filtered_by_level", "filtered_by_shard", "missing_input", "missing_control", "invalid_rows", "valid"]:
        print(f"  {key}: {stats[key]}")
    print(f"  task_counts: {stats['task_counts']}")
    print(f"  control_counts: {stats['control_counts']}")
    if stats["missing_input_examples"]:
        print(f"  missing_input_examples: {stats['missing_input_examples']}")
    if stats["missing_control_examples"]:
        print(f"  missing_control_examples: {stats['missing_control_examples']}")
    if spec.get("control_mode") == "source_camera_trajectory":
        print(f"  source_pair_key_mismatch: {stats['source_pair_key_mismatch']}")
        if stats["source_pair_key_mismatch_examples"]:
            print(f"  source_pair_key_mismatch_examples: {stats['source_pair_key_mismatch_examples']}")
    if not samples:
        raise SystemExit("No valid samples after filtering; dry-run failed.")
    path_errors = validate_model_paths(model_key, args) if args.check_model_paths else []
    if path_errors:
        raise SystemExit("Model path validation failed: " + "; ".join(path_errors))
    preview = list(samples[: max(args.print_samples, 0)])
    config = build_config(model_key, args, preview)
    print("Dry-run config preview:")
    print(json.dumps({k: v for k, v in config.items() if k != "samples"}, indent=2, ensure_ascii=True))
    for sample in preview:
        print("-" * 80)
        print(f"sample_id: {sample['sample_id']}")
        print(f"task: {sample['task']} control_type: {sample['control_type']}")
        print(f"first_frame_path: {sample['first_frame_path']}")
        if spec.get("control_mode") == "keyboard_mouse":
            print(f"metadata_control_txt_path: {sample['metadata_control_txt_path']}")
            print("matrixgame_control:")
            print(json.dumps(sample["matrixgame_control"], indent=2, ensure_ascii=True))
        elif spec.get("control_mode") == "text_prompt":
            print(f"metadata_control_txt_path: {sample['metadata_control_txt_path']}")
            print("text_control:")
            print(json.dumps(sample["text_control"], indent=2, ensure_ascii=True))
            print("prompt_csv_row:")
            print(json.dumps({"Image Filename": prompt_csv_image_filename(sample, args), "Prompt": sample["prompt"]}, indent=2, ensure_ascii=True))
            if model_key == "yume":
                if sample["task"].lower() == "mem":
                    print("yume_adapter_csv_row:")
                    print(json.dumps(build_yume_memory_csv_rows([sample])[0], indent=2, ensure_ascii=True))
                else:
                    print("yume_adapter_csv_row:")
                    print(json.dumps(build_yume_action_csv_rows([sample])[0], indent=2, ensure_ascii=True))
        elif spec.get("control_mode") == "source_camera_trajectory":
            print(f"source_video_filename: {sample.get('source_video_filename')}")
            print(f"source_camera_txt_path: {sample.get('source_camera_txt_path')}")
            print(f"source_pair_key: {sample.get('source_pair_key')}")
            print(f"source_pair_key_matches: {sample.get('source_pair_key_matches')}")
        else:
            print(f"control_txt_path: {sample['control_txt_path']}")
        print("constructed_command:")
        print("  " + shlex.join(build_command(model_key, args, sample)))
    if args.write_manifest:
        manifest = {
            "model": model_key,
            "stats": stats,
            "config": build_config(model_key, args, samples),
            "commands": [build_command(model_key, args, sample) for sample in samples],
        }
        manifest_path = Path(args.write_manifest).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        print(f"Wrote dry-run manifest: {manifest_path}")
        if spec.get("control_mode") == "text_prompt":
            prompt_csv_path = Path(text_prompt_csv_path(model_key, args)).expanduser()
            write_csv_rows(prompt_csv_path, build_prompt_csv_rows(samples, args), ["Image Filename", "Prompt"])
            print(f"Wrote text prompt CSV: {prompt_csv_path}")
            if model_key == "yume":
                action_rows = build_yume_action_csv_rows(samples)
                memory_rows = build_yume_memory_csv_rows(samples)
                if action_rows:
                    action_path = Path(yume_adapter_csv_path(args, "Diff")).expanduser()
                    write_csv_rows(action_path, action_rows, ["\u6587\u4ef6\u540d", "\u7ea7\u522b", "\u5e73\u52a8", "\u8f6c\u52a8"])
                    print(f"Wrote YUME action CSV: {action_path}")
                if memory_rows:
                    memory_path = Path(yume_adapter_csv_path(args, "Mem")).expanduser()
                    write_csv_rows(memory_path, memory_rows, ["\u6587\u4ef6\u540d", "\u5e8f\u53f7"])
                    print(f"Wrote YUME memory CSV: {memory_path}")
    print("Dry-run completed without executing inference.")


def main(model_key: str) -> int:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model demo: {model_key}")
    spec = MODEL_SPECS[model_key]
    parser = argparse.ArgumentParser(description=f"Dry-run iWorldBench inference demo for {spec['display_name']}")
    add_common_args(parser, model_key)
    args = parser.parse_args()
    samples, stats = build_samples(args, spec)
    print_report(model_key, args, samples, stats)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: demo_common.py <model_key> [args]")
    model = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    raise SystemExit(main(model))
