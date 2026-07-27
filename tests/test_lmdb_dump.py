"""Tests for LMDB frame dump functionality.

Covers:
- pts helper functions (pts_difference_per_frame, frame_index_to_pts)
- PyAVSTADataset.__init__: chunk building, existing-key filtering, video_uid filter
- PyAVSTADataset.__getitem__: retry logic, key format, missing-frame warnings
- collate_fn: batch merging
- Ego4DHLMDB: put/get roundtrip, put_batch/get_batch, get_existing_keys, custom template

Reference: https://github.com/EGO4D/forecasting/blob/main/tools/short_term_anticipation/dump_frames_to_lmdb_files.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add script directory so PyAVSTADataset / collate_fn can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "script"))

from dump_frame_to_lmdb_files import collate_fn, PyAVSTADataset

from sta_baseline.datasets.short_term_anticipation import (
    Ego4DHLMDB,
    frame_index_to_pts,
    pts_difference_pre_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_FORMAT = "{video_id:s}_{frame_number:07d}"


def _make_annotation(video_uid: str, frame: int) -> dict:
    return {"video_uid": video_uid, "frame": frame}


def _make_dataset(
    annotations: list[dict],
    *,
    existing_keys: list[bytes] | None = None,
    video_uid: list[str] | None = None,
    context_frames: int = 4,
    max_chunk_size: int = 32,
    fname_format: str = _FRAME_FORMAT,
) -> PyAVSTADataset:
    """Helper to create a PyAVSTADataset without real video files."""
    ds = PyAVSTADataset.__new__(PyAVSTADataset)
    # Call __init__ directly with a fake path (no videos are opened here)
    PyAVSTADataset.__init__(
        ds,
        video_uid=video_uid,
        annotations=annotations,
        path_to_videos=Path("/fake/videos"),
        existing_keys=existing_keys or [],
        context_frames=context_frames,
        max_chunk_size=max_chunk_size,
        fname_format=fname_format,
    )
    return ds


# ---------------------------------------------------------------------------
# pts helper functions
# ---------------------------------------------------------------------------


class TestPtsDifferencePerFrame:
    def test_30fps_90000_timebase(self) -> None:
        from fractions import Fraction

        fps = Fraction(30)
        time_base = Fraction(1, 90000)
        assert pts_difference_pre_frame(fps, time_base) == 3000

    def test_25fps_12800_timebase(self) -> None:
        from fractions import Fraction

        fps = Fraction(25)
        time_base = Fraction(1, 12800)
        assert pts_difference_pre_frame(fps, time_base) == 512

    def test_24fps_standard(self) -> None:
        from fractions import Fraction

        fps = Fraction(24)
        time_base = Fraction(1, 24000)
        assert pts_difference_pre_frame(fps, time_base) == 1000


class TestFrameIndexToPts:
    def test_frame_zero_returns_start(self) -> None:
        assert frame_index_to_pts(0, start_pts=0, diff_per_frame=3000) == 0

    def test_frame_five(self) -> None:
        assert frame_index_to_pts(5, start_pts=0, diff_per_frame=3000) == 15000

    def test_nonzero_start(self) -> None:
        assert frame_index_to_pts(5, start_pts=100, diff_per_frame=3000) == 15100

    def test_frame_one(self) -> None:
        assert frame_index_to_pts(1, start_pts=0, diff_per_frame=512) == 512


# ---------------------------------------------------------------------------
# PyAVSTADataset.__init__ - chunk building
# ---------------------------------------------------------------------------


class TestPyAVSTADatasetInit:
    def test_single_annotation_creates_one_chunk(self) -> None:
        ds = _make_dataset([_make_annotation("vid_a", frame=3)], context_frames=4)
        # frames 0,1,2,3 - consecutive -> 1 chunk
        assert len(ds.chunks) == 1
        video_id, chunk = ds.chunks[0]
        assert video_id == "vid_a"
        np.testing.assert_array_equal(chunk, [0, 1, 2, 3])

    def test_first_frame_clamped_to_zero(self) -> None:
        ds = _make_dataset([_make_annotation("vid_a", frame=1)], context_frames=4)
        _, chunk = ds.chunks[0]
        assert chunk[0] == 0

    def test_existing_keys_are_skipped(self) -> None:
        # frames 0..3; key for frame 1 already exists → frames 0,2,3 remain
        key = f"vid_a_{1:07d}".encode()
        ds = _make_dataset(
            [_make_annotation("vid_a", frame=3)],
            existing_keys=[key],
            context_frames=4,
        )
        all_frames = np.concatenate([c for _, c in ds.chunks])
        assert 1 not in all_frames
        assert set(all_frames) == {0, 2, 3}

    def test_non_consecutive_frames_split_into_multiple_chunks(self) -> None:
        # Two annotations for the same video, far apart → 2 consecutive groups
        ann = [
            _make_annotation("vid_a", frame=3),   # frames 0-3
            _make_annotation("vid_a", frame=10),  # frames 7-10 (context_frames=4)
        ]
        ds = _make_dataset(ann, context_frames=4)
        assert len(ds.chunks) >= 2

    def test_video_uid_filter(self) -> None:
        anns = [
            _make_annotation("vid_a", frame=5),
            _make_annotation("vid_b", frame=5),
        ]
        ds = _make_dataset(anns, video_uid=["vid_b"], context_frames=4)
        assert all(vid == "vid_b" for vid, _ in ds.chunks)

    def test_chunk_split_when_exceeds_max_chunk_size(self) -> None:
        # One annotation with many consecutive frames
        ann = [_make_annotation("vid_a", frame=100)]
        ds = _make_dataset(ann, context_frames=20, max_chunk_size=8)
        for _, chunk in ds.chunks:
            assert len(chunk) <= 8

    def test_len_equals_number_of_chunks(self) -> None:
        ds = _make_dataset([_make_annotation("vid_a", frame=3)], context_frames=4)
        assert len(ds) == len(ds.chunks)

    def test_underscored_video_id_in_existing_key(self) -> None:
        """rsplit() must handle video IDs that contain underscores."""
        key = f"vid_long_id_{5:07d}".encode()
        ds = _make_dataset(
            [_make_annotation("vid_long_id", frame=6)],
            existing_keys=[key],
            context_frames=4,
        )
        all_frames = np.concatenate([c for _, c in ds.chunks])
        assert 5 not in all_frames


# ---------------------------------------------------------------------------
# PyAVSTADataset.__getitem__ - retry logic and key format
# ---------------------------------------------------------------------------


class TestPyAVSTADatasetGetItem:
    def _make_getitem_dataset(
        self,
        video_id: str,
        frame_numbers: list[int],
        fname_format: str = _FRAME_FORMAT,
    ) -> PyAVSTADataset:
        ds = object.__new__(PyAVSTADataset)
        ds.chunks = [(video_id, np.array(frame_numbers))]
        ds.path_to_videos = Path("/fake/videos")
        ds.retry = 3
        ds.frame_height = 320
        ds.fname_format = fname_format
        return ds

    def _fake_image(self) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def test_all_frames_retrieved_first_try(self) -> None:
        ds = self._make_getitem_dataset("vid_a", [0, 1, 2])
        imgs = [self._fake_image(), self._fake_image(), self._fake_image()]

        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(return_value=imgs)

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", return_value=mock_vr):
            result = ds[0]

        assert len(result["ims"]) == 3
        assert len(result["keys"]) == 3

    def test_keys_use_fname_format(self) -> None:
        fmt = "{video_id:s}_{frame_number:010d}"
        ds = self._make_getitem_dataset("vid_a", [0, 1], fname_format=fmt)
        imgs = [self._fake_image(), self._fake_image()]

        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(return_value=imgs)

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", return_value=mock_vr):
            result = ds[0]

        assert result["keys"][0] == "vid_a_0000000000"
        assert result["keys"][1] == "vid_a_0000000001"

    def test_retry_recovers_missing_frames(self) -> None:
        ds = self._make_getitem_dataset("vid_a", [0, 1, 2])
        img = self._fake_image()

        # First call: only frame 0 succeeds (None for 1, 2)
        # Second call: all succeed
        call_count = [0]

        def fake_vr_getitem(_frames: np.ndarray) -> list[np.ndarray | None]:
            call_count[0] += 1
            if call_count[0] == 1:
                return [img, None, None]
            return [img, img]  # frames 1,2 on retry

        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(side_effect=fake_vr_getitem)

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", return_value=mock_vr):
            result = ds[0]

        assert len(result["ims"]) == 3

    def test_missing_frames_not_in_output(self) -> None:
        ds = self._make_getitem_dataset("vid_a", [0, 1, 2])
        img = self._fake_image()

        # Frame 1 always fails regardless of how many frames are requested
        def fake_getitem(frames: np.ndarray) -> list[np.ndarray | None]:
            return [img if int(f) != 1 else None for f in frames]

        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(side_effect=fake_getitem)

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", return_value=mock_vr):
            result = ds[0]

        assert len(result["ims"]) == 2
        assert "vid_a_0000001" not in result["keys"]

    def test_warning_printed_for_missing_frames(self, capsys: pytest.CaptureFixture) -> None:
        ds = self._make_getitem_dataset("vid_a", [0, 1])
        img = self._fake_image()

        def fake_getitem(frames: np.ndarray) -> list[np.ndarray | None]:
            return [img if int(f) != 1 else None for f in frames]

        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(side_effect=fake_getitem)

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", return_value=mock_vr):
            ds[0]

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_ims_and_keys_same_length(self) -> None:
        ds = self._make_getitem_dataset("vid_a", [0, 1, 2, 3])
        imgs = [self._fake_image()] * 4

        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(return_value=imgs)

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", return_value=mock_vr):
            result = ds[0]

        assert len(result["ims"]) == len(result["keys"])

    def test_early_exit_when_all_frames_collected(self) -> None:
        """PyAVVideoReader should only be called once when all frames succeed."""
        ds = self._make_getitem_dataset("vid_a", [0, 1])
        mock_vr = MagicMock()
        mock_vr.__getitem__ = MagicMock(return_value=[self._fake_image(), self._fake_image()])

        constructor_calls: list[int] = []

        def _side_effect(*_a: object, **_kw: object) -> MagicMock:
            constructor_calls.append(1)
            return mock_vr

        with patch("dump_frame_to_lmdb_files.PyAVVideoReader", side_effect=_side_effect):
            ds[0]

        # Should stop after 1 successful attempt
        assert len(constructor_calls) == 1


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def _fake_image(self) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def test_merges_single_sample(self) -> None:
        batch = [{"ims": [self._fake_image()], "keys": ["vid_a_0000001"]}]
        result = collate_fn(batch)
        assert len(result["ims"]) == 1
        assert result["keys"] == ["vid_a_0000001"]

    def test_merges_multiple_samples(self) -> None:
        batch = [
            {"ims": [self._fake_image(), self._fake_image()], "keys": ["k0", "k1"]},
            {"ims": [self._fake_image()], "keys": ["k2"]},
        ]
        result = collate_fn(batch)
        assert len(result["ims"]) == 3
        assert result["keys"] == ["k0", "k1", "k2"]

    def test_output_keys_match_ims_length(self) -> None:
        batch = [
            {"ims": [self._fake_image()] * 5, "keys": [f"k{i}" for i in range(5)]},
        ]
        result = collate_fn(batch)
        assert len(result["ims"]) == len(result["keys"])


# ---------------------------------------------------------------------------
# Ego4DHLMDB
# ---------------------------------------------------------------------------


class TestEgo4DHLMDB:
    def _random_bgr(self, h: int = 32, w: int = 32) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_put_then_get_returns_image(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        img = self._random_bgr()
        db.put("vid_a", 0, img)
        result = db.get("vid_a", 0)
        assert result is not None
        assert result.shape[:2] == img.shape[:2]

    def test_put_batch_then_get_batch_roundtrip(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        imgs = [self._random_bgr() for _ in range(3)]
        db.put_batch("vid_a", [0, 1, 2], imgs)
        results = db.get_batch("vid_a", [0, 1, 2])
        assert len(results) == 3
        for r, orig in zip(results, imgs, strict=True):
            assert r is not None
            assert r.shape[:2] == orig.shape[:2]

    def test_get_nonexistent_frame_returns_none(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        db.put("vid_a", 0, self._random_bgr())
        # Frame 99 was never written
        result = db.get("vid_a", 99)
        assert result is None

    def test_get_batch_missing_frame_returns_none(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        db.put("vid_a", 0, self._random_bgr())
        results = db.get_batch("vid_a", [0, 1])  # frame 1 missing
        assert results[0] is not None
        assert results[1] is None

    def test_get_existing_keys_returns_bytes(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        db.put("vid_a", 5, self._random_bgr())
        keys = db.get_existing_keys()
        assert len(keys) == 1
        assert isinstance(keys[0], bytes)

    def test_get_existing_keys_after_put_batch(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        db.put_batch("vid_a", [0, 1, 2], [self._random_bgr() for _ in range(3)])
        keys = db.get_existing_keys()
        assert len(keys) == 3

    def test_get_existing_keys_multiple_videos(self, tmp_path: Path) -> None:
        db = Ego4DHLMDB(tmp_path)
        db.put("vid_a", 0, self._random_bgr())
        db.put("vid_b", 0, self._random_bgr())
        keys = db.get_existing_keys()
        assert len(keys) == 2

    def test_custom_frame_template(self, tmp_path: Path) -> None:
        template = "{video_id:s}_{frame_number:05d}"
        db = Ego4DHLMDB(tmp_path, frame_template=template)
        img = self._random_bgr()
        db.put("vid_a", 7, img)
        result = db.get("vid_a", 7)
        assert result is not None

    def test_existing_keys_parseable_with_rsplit(self, tmp_path: Path) -> None:
        """Keys produced by put must be parseable with rsplit('_', 1)."""
        db = Ego4DHLMDB(tmp_path)
        db.put("vid_with_underscores", 3, self._random_bgr())
        keys = db.get_existing_keys()
        assert len(keys) == 1
        video_id, frame_str = keys[0].decode().rsplit("_", 1)
        assert video_id == "vid_with_underscores"
        assert int(frame_str) == 3
