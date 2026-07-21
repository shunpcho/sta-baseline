import io
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

import av
import imutils
import lmdb
import numpy as np
import numpy.typing as npt
from cv2 import imdecode, imencode, IMREAD_COLOR

FrameList: TypeAlias = int | float | list[int | float] | tuple[int | float, ...]


def _get_frames(
    frame_list: list[int],
    container: av.container.Container,
    include_audio: bool = False,
    audio_buffer_frames: int = 0,
) -> Iterable[av.frame.Frame]:
    if len(container.streams.video) == 0:
        raise ValueError(f"No video streams found in {container.name}")
    if len(container.streams.video) > 1:
        raise ValueError(f"Multiple video streams not supported in {container.name}")

    video_stream = container.streams.video[0]
    video_start = video_stream.start_time
    video_base = video_stream.time_base
    fps = video_stream.average_rate
    video_pt_diff = pts_difference_pre_frame(fps, video_base)

    audio_buffer_pts = frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff) if include_audio else 0


def pts_difference_pre_frame(fps: Fraction, time_base: Fraction) -> int: ...


def frame_index_to_pts(frame: int, start_pts: int, diff_per_frame: int) -> int:
    return start_pts + frame * diff_per_frame


class PyAVVideoReader:
    """To read frames from a video file using PyAV."""

    def __init__(
        self, path_to_video: Path, include_audio: bool = False, audio_buffer_frames: int = 0, height: int | None = None
    ) -> None:
        """Initialize the PyAVVideoReader.

        Args:
            path_to_video: Path to the video file.
            include_audio: Whether to include audio frames.
            audio_buffer_frames: Number of audio frames to buffer.
            height: Desired height of the output frames.
        """
        self.path_to_video = path_to_video
        self.include_audio = include_audio
        self.audio_buffer_frames = audio_buffer_frames
        self.height = height

    def __getitem__(self, frame_list: FrameList) -> list[np.ndarray]:
        """Get frames from the video based on the provided frame list.

        Args:
            frame_list: List of frame indices to retrieve.

        Returns:
            List of frames as numpy arrays. If a frame is not available, None is returned.
        """
        if isinstance(frame_list, (int, float)):
            frame_list = [int(frame_list)]
        elif not isinstance(frame_list, (list, tuple)):
            frame_list = [int(frame) for frame in frame_list]
        else:
            frame_list = list(frame_list)

        with av.open(self.path_to_video) as input_video:
            frames = _get_frames(
                frame_list, input_video, include_audio=self.include_audio, audio_buffer_frames=self.audio_buffer_frames
            )
            frames = list(frames)
        frames = [f.to_ndarray(format="rgb24") if f is not None else None for f in frames]

        if self.height is not None:
            frames = [imutils.resize(f, height=self.height) if f is not None else None for f in frames]
        return frames


class Ego4DHLMDB:
    def __init__(
        self,
        path_to_root: Path,
        readonly: bool = False,
        lock: bool = False,
        frame_template: str | None = None,
        map_size: int = 1099511627776,
    ) -> None:
        """Initialize the Ego4DHLMDB.

        Args:
            path_to_root: Path to the root directory containing LMDB files.
            readonly: Whether to open the LMDBs in read-only mode.
            lock: Whether to use locking when accessing the LMDBs.
            frame_template: Template for frame keys in the LMDBs.
            map_size: Maximum size of the LMDBs in bytes.
        """
        self.environments = {}
        self.path_to_root = path_to_root
        self.path_to_root.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.lock = lock
        self.map_size = map_size
        self.frame_template = frame_template or "{video_id:s}_{frame_number:010d}"

    def _get_parent(self, parent: str) -> lmdb.Environment:
        """Get or create an LMDB environment for the specified parent."""
        return lmdb.open(
            str(self.path_to_root / parent), map_size=self.map_size, readonly=self.readonly, lock=self.lock
        )

    def put_batch(self, video_id: str, frames: list[int], data: list[npt.NDArray[np.uint8]]) -> None:
        with self._get_parent(video_id) as env, env.begin(write=True) as txn:
            for frame_number, frame_data in zip(frames, data, strict=True):
                txn.put(
                    self.frame_template.format(video_id=video_id, frame_number=frame_number).encode(),
                    imencode(".jpg", frame_data)[1],
                )

    def put(self, video_id: str, frame: int, data: npt.NDArray[np.uint8]) -> None:
        with self._get_parent(video_id) as env, env.begin(write=True) as txn:
            txn.put(
                self.frame_template.format(video_id=video_id, frame_number=frame).encode(),
                imencode(".jpg", data)[1],
            )

    def get(self, video_id: str, frame: int) -> npt.NDArray[np.uint8] | None:
        """Get a frame from the LMDB for the specified video ID and frame number.

        Args:
            video_id: The unique identifier of the video.
            frame: The frame number to retrieve.

        Returns:
            The frame as a numpy array.
        """
        with self._get_parent(video_id) as env, env.begin(write=False) as txn:
            data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())

            file_bytes = np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.uint8) if data is not None else None
            return imdecode(file_bytes, IMREAD_COLOR)

    def get_batch(self, video_id: str, frames: list[int]) -> list[npt.NDArray[np.uint8] | None]:
        """Get a batch of frames from the LMDB for the specified video ID and frame numbers.

        Args:
            video_id: The unique identifier of the video.
            frames: List of frame numbers to retrieve.

        Returns:
            List of frames as numpy arrays. If a frame is not available, None is returned.
        """
        out: list[npt.NDArray[np.uint8] | None] = []
        with self._get_parent(video_id) as env, env.begin(write=False) as txn:
            for frame in frames:
                data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())
                file_bytes = (
                    np.asarray(bytearray(io.BytesIO(data).read()), dtype=np.uint8) if data is not None else None
                )
                out.append(imdecode(file_bytes, IMREAD_COLOR) if file_bytes is not None else None)
        return out

    def get_existing_keys(self) -> list[lmdb.Cursor]:
        """Get a list of existing keys in the LMDB.

        Returns:
            List of existing keys as strings.
        """
        existing_keys: list[lmdb.Cursor] = []
        for parent in self.path_to_root.iterdir():
            if parent.is_dir():
                with self._get_parent(parent.name) as env, env.begin(write=False) as txn:
                    existing_keys += list(txn.cursor().iternext(values=False))
        return existing_keys
