from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import DataLoader

from scripts.dump_frame_to_lmdb_files import collate_fn, LMDBAnnotation, PyAVSTADataset
from sta_baseline.datasets.short_term_anticipation import Ego4DHLMDB, Ego4dShortTermAnticipation, PyAVVideoReader
from tests.conftest import create_dummy_video


@pytest.fixture
def dummy_videos(tmp_path: Path, video_names: list[str]) -> list[Path]:
    """Create dummy video files in a temporary directory."""
    video_paths = []
    for video_name in video_names:
        video_path = tmp_path.parent / "videos" / video_name
        video_path.parent.mkdir(parents=True, exist_ok=True)
        create_dummy_video(video_path, frames=20)
        video_paths.append(video_path)
    return video_paths


@pytest.mark.parametrize("video_name", ["dummy162435.mp4"])
def test_dataset(dummy_videos: list[Path], video_name: str) -> None:
    """Test that PyAVSTADataset can be instantiated with dummy video."""
    dummy_annotations: list[LMDBAnnotation] = [
        {
            "clip_uid": "dummy162435",
            "clip_frame": 10,
        }
    ]
    ds = PyAVSTADataset(
        clip_uid=None,
        annotations=dummy_annotations,
        path_to_videos=dummy_videos[0].parent,
        existing_keys=[],
        context_frames=4,
        max_chunk_size=4,
        frame_height=96,
        fname_format="{video_id:s}_{frame_number:07d}",
    )
    assert ds.path_to_videos.exists()

    video_id, frame_numbers = ds.chunks[0]
    assert video_id == "dummy162435"
    assert (frame_numbers == np.array([7, 8, 9, 10])).all()
    assert len(ds) == 1

    assert len(ds.chunks[0][1]) == 4
    assert (np.setdiff1d(frame_numbers, 0) == np.array([7, 8, 9, 10])).all()

    sample = ds[0]
    assert len(sample["ims"]) == 4
    assert sample["keys"] == [
        "dummy162435_0000007",
        "dummy162435_0000008",
        "dummy162435_0000009",
        "dummy162435_0000010",
    ]


@pytest.mark.parametrize("video_name", ["dummy4587.mp4"])
def test_video_reader(dummy_videos: list[Path], video_name: str) -> None:
    """Test that PyAVVideoReader can read frames from the dummy video."""
    reader = PyAVVideoReader(str(dummy_videos[0]), height=96)
    ims = reader[np.array([7, 8, 9, 10])]
    assert len(ims) == 4
    for frame in ims:
        assert frame.shape == (96, 128, 3)
    assert ims[0].dtype == np.uint8


@pytest.mark.parametrize("video_name", ["dummy4587.mp4"])
def test_ego4d_h_lmdb(dummy_videos: list[Path], video_name: str) -> None:
    """Test that Ego4DHLMDB can be instantiated with dummy video."""
    lmdb_dataset = Ego4DHLMDB(path_to_root=dummy_videos[0].parent)

    existing_keys = lmdb_dataset.get_existing_keys()
    # If there are lmdb files, the existing keys should not be empty.
    # In this test, we expect the lmdb files to be empty, so the existing keys should be empty.
    assert len(existing_keys) == 0


def test_load_detections_allows_missing_uid() -> None:
    """Missing detection entries are treated as samples with no detections."""
    dataset = object.__new__(Ego4dShortTermAnticipation)
    dataset._obj_detections = {}
    dataset.cfg = SimpleNamespace(EGO4D_STA=SimpleNamespace(DETECTION_SCORE_THRESH=0.5))

    pred_boxes, pred_object_labels, pred_scores = dataset._load_detections("missing-uid")

    assert pred_boxes.shape == (0, 4)
    assert pred_object_labels.size == 0
    assert pred_scores.size == 0


@pytest.mark.parametrize("video_name", ["dummy4587.mp4"])
def test_dataloader(dummy_videos: list[Path], video_name: str) -> None:
    """Test that the dataloader can be created with dummy video."""
    dummy_annotations: list[LMDBAnnotation] = [
        {
            "clip_uid": "dummy4587",
            "clip_frame": 10,
        }
    ]
    ds = PyAVSTADataset(
        clip_uid=None,
        annotations=dummy_annotations,
        path_to_videos=dummy_videos[0].parent,
        existing_keys=[],
        context_frames=4,
        max_chunk_size=4,
        frame_height=96,
        fname_format="{video_id:s}_{frame_number:07d}",
    )
    dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    batch = next(iter(dataloader))
    frames, keys = batch["ims"], batch["keys"]
    assert len(frames) == 4
    assert len(keys) == 4


@pytest.mark.parametrize("video_name", ["dummy4587.mp4"])
def test_create_lmdb(dummy_videos: list[Path], video_name: str) -> None:
    """Test that the lmdb can be created with dummy video."""
    dummy_annotations: list[LMDBAnnotation] = [
        {
            "clip_uid": "dummy4587",
            "clip_frame": 10,
        }
    ]

    lmdb_dataset = Ego4DHLMDB(path_to_root=dummy_videos[0].parent)

    ds = PyAVSTADataset(
        clip_uid=None,
        annotations=dummy_annotations,
        path_to_videos=dummy_videos[0].parent,
        existing_keys=[],
        context_frames=4,
        max_chunk_size=4,
        frame_height=96,
        fname_format="{video_id:s}_{frame_number:07d}",
    )
    dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    # Test creating lmdb files and lmdb not empty.
    batch = next(iter(dataloader))
    frames, keys = batch["ims"], batch["keys"]
    for parent in np.unique([k.rsplit("_", 1)[0] for k in keys]):
        idx = np.where([k.startswith(parent + "_") for k in keys])[0]
        these_keys = [int(keys[i].rsplit("_", 1)[1]) for i in idx]
        these_frames = [frames[i] for i in idx]
        lmdb_dataset.put_batch(parent, these_keys, these_frames)
    assert len(these_keys) == 4
    assert (these_keys == np.array([7, 8, 9, 10])).all()
    assert all(frame.shape == (96, 128, 3) for frame in these_frames)
    assert dummy_videos[0].with_name(dummy_videos[0].stem).joinpath("data.mdb").exists()

    # Check that the lmdb can be read the contents.
    contents = lmdb_dataset.get_batch("dummy4587", [7, 8, 9, 10])
    assert len(contents) == 4
    for frame in contents:
        assert frame.shape == (96, 128, 3)
