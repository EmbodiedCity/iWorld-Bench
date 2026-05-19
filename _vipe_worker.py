#!/usr/bin/env python3
"""
Standalone VIPe worker — called via subprocess.Popen by unified_video_metrics._run_vipe().
Args: <videos_json_file> <vipe_output_dir> <process_index>
CUDA_VISIBLE_DEVICES must be set by the caller before launching.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_VIPE_ROOT = os.environ.get("VIPE_ROOT") or str(_THIS_DIR / "third_party" / "vipe")
sys.path.append(_VIPE_ROOT)

POSE_DIR_NAME = "pose"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _is_labeled(video_path: str, vipe_output_dir: str) -> bool:
    stem = Path(video_path).stem
    return (Path(vipe_output_dir) / POSE_DIR_NAME / f"{stem}.npz").exists()


def main():
    videos_json_file = sys.argv[1]
    vipe_output_dir = sys.argv[2]
    proc_idx = int(sys.argv[3])

    logger = logging.getLogger(f"vipe.worker{proc_idx}")

    with open(videos_json_file) as f:
        video_paths = json.load(f)

    # Disable HuggingFace network calls and remove any proxy that might block loading
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for _proxy_key in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ.pop(_proxy_key, None)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    logger.info(f"Worker {proc_idx} on GPU {gpu_id}: {len(video_paths)} videos assigned")

    import hydra
    from vipe import get_config_path
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.streams.base import ProcessedVideoStream
    from vipe.pipeline import make_pipeline as vipe_make_pipeline

    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already restricts to one GPU
    torch.cuda.empty_cache()
    time.sleep(proc_idx * 2)  # stagger startup to reduce model-load contention

    unlabeled = [v for v in video_paths if not _is_labeled(v, vipe_output_dir)]
    logger.info(f"Worker {proc_idx}: {len(unlabeled)} unlabeled, {len(video_paths) - len(unlabeled)} already done")

    for video_path in unlabeled:
        video_name = Path(video_path).name
        try:
            if _is_labeled(video_path, vipe_output_dir):
                logger.info(f"Worker {proc_idx}: skip {video_name} (NPZ exists)")
                continue

            torch.cuda.empty_cache()

            overrides = [
                "pipeline=default",
                f"pipeline.output.path={vipe_output_dir}",
                "pipeline.output.save_artifacts=true",
                "pipeline.output.save_viz=false",
                "pipeline.slam.visualize=false",
            ]
            with hydra.initialize_config_dir(
                config_dir=str(get_config_path()), version_base=None
            ):
                cfg = hydra.compose(
                    config_name="default",
                    overrides=overrides,
                    return_hydra_config=False,
                )

            stream = ProcessedVideoStream(
                RawMp4Stream(Path(video_path)), []
            ).cache(desc="Reading video stream")
            vipe_make_pipeline(cfg.pipeline).run(stream)

            # Remove non-pose artefacts to save disk space
            stem = Path(video_path).stem
            for entry in Path(vipe_output_dir).iterdir():
                if entry.is_dir() and entry.name != POSE_DIR_NAME:
                    for f in entry.glob(f"{stem}*"):
                        f.unlink(missing_ok=True)

            logger.info(f"Worker {proc_idx}: done {video_name}")

        except Exception as exc:
            logger.error(f"Worker {proc_idx}: failed {video_name}: {str(exc)[:300]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
