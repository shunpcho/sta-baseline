"""Shared pytest configuration and fixtures.

Provides a minimal torch.utils.data.Dataset stub so that the dump script can be
imported in CI environments where PyTorch is not installed.  When torch is already
available the real module is used unchanged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.dump_frame_to_lmdb_files import LMDBAnnotation


def create_dummy_video(movie_path: Path, frames: int = 40) -> None:
    """Return a dummy video frame for testing."""
    writer = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 96))

    for _ in range(frames):  # 1 second of video at 30 fps
        frame = np.random.randint(0, 256, (96, 128, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def create_dummy_annotations() -> list[LMDBAnnotation]:
    """Create dummy annotations for testing."""
    annotations: LMDBAnnotation = {
        "video_uid": "__",
        "frame": 1136,
        "clip_uid": "dummy_video",
        "clip_frame": 2,
    }
    return [annotations]
