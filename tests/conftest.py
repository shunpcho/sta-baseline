"""Shared pytest configuration and fixtures.

Provides a minimal torch.utils.data.Dataset stub so that the dump script can be
imported in CI environments where PyTorch is not installed.  When torch is already
available the real module is used unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import cv2
import lmdb
import numpy as np

from sta_baseline.utils.type_alias import (
    FHOSTAInfo,
    ObjectDetectionAnnotation,
    ObjectsAnnotation,
)


def create_dummy_video(movie_path: Path, frames: int = 40) -> None:
    """Return a dummy video frame for testing."""
    writer = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 96))

    for _ in range(frames):  # 1 second of video at 30 fps
        frame = np.random.randint(0, 256, (96, 128, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def create_dummy_json_object_detections(root_path: Path, object_detection: ObjectDetectionAnnotation) -> Path:
    """Create dummy object detections for testing."""
    object_json_path = root_path / "object_detections.json"
    with object_json_path.open("w", encoding="utf-8") as f:
        json.dump(object_detection, f)
    return object_json_path


def create_dummy_lmdb(root_path: Path, video_id: str, frame_numbers: list[int], image_size: tuple[int, int]) -> Path:
    """Create a dummy LMDB directory for testing.

    Args:
        root_path: The root path where the LMDB directory will be created.
        video_id: The video ID to use for the LMDB. LMDB path will be "{video_id}/data.mdb".
        frame_numbers: The frame numbers to include in the LMDB.
        image_size: The size of the images to create in the LMDB, as (width, height).

    Returns:
        The path to the created LMDB directory.
    """
    these_keys = np.array(frame_numbers)
    these_frames = [
        np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8) for _ in range(len(these_keys))
    ]

    lmdb_path = root_path / f"{video_id}/data.mdb"
    with (
        lmdb.open(str(lmdb_path.parent), map_size=1 << 40, readonly=False, lock=False) as env,
        env.begin(write=True) as txn,
    ):
        for frame_number, frame_data in zip(these_keys, these_frames, strict=True):
            txn.put(f"{video_id}_{frame_number:07d}".encode(), cv2.imencode(".jpg", frame_data)[1])
    return lmdb_path.parent


class DummyFHOSTA(TypedDict):
    info: FHOSTAInfo
    annotations: list[DummyAnnotation]


class DummyAnnotation(TypedDict):
    uid: str
    main_uid: str
    video_uid: str
    frame: int
    clip_id: int
    clip_uid: str
    clip_frame: int
    objects: list[ObjectsAnnotation]


def create_dummy_sta_train_json(root_path: Path, sta_train_annotations: DummyFHOSTA) -> Path:
    """Create a dummy STA train JSON file for testing."""
    sta_train_json_path = root_path / "fho_sta_train.json"
    with sta_train_json_path.open("w", encoding="utf-8") as f:
        json.dump(sta_train_annotations, f)
    return sta_train_json_path
