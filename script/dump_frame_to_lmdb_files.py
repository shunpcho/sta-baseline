"""Script to dump clip video frames into LMDBs for the Short-Term Anticipation task.

Structure of the output LMDBs:
lmdb/
  video_00001/
    key: "<video_id>_<frame_idx>"
    value: JPEG/PNG bytes of the frame image
"""

import json
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sta_baseline.datasets.short_term_anticipation import Ego4DHLMDB, PyAVVideoReader


def main() -> None:
    """Main function to parse command-line arguments and create LMDBs."""
    parser = ArgumentParser()

    parser.add_argument("path_to_annotations", type=Path, help="Path to the annotations file.")
    parser.add_argument("path_to_videos", type=Path, help="Path to the directory containing the video files.")
    parser.add_argument(
        "path_to_output_lmdbs", type=Path, help="Path to the directory where the output LMDBs will be stored."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing videos.")
    parser.add_argument("--context_frames", type=int, default=32, help="Number of context frames to use.")
    parser.add_argument(
        "--fname_format", type=str, default="{video_id:s}_{frame_number:07d}", help="Format for the frame filenames."
    )
    parser.add_argument("--frame_height", type=int, default=320, help="Height of the video frames.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for the DataLoader.")
    parser.add_argument("--video_uid", type=str, nargs="+", default=None, help="Unique identifier(s) for the video(s).")

    args = parser.parse_args()

    with Path(args.path_to_annotations / "fho_sta_train.json").open(encoding="utf-8") as f:
        train = json.load(f)
    with Path(args.path_to_annotations / "fho_sta_val.json").open(encoding="utf-8") as f:
        val = json.load(f)
    with Path(args.path_to_annotations / "fho_sta_test_unannotated.json").open(encoding="utf-8") as f:
        test = json.load(f)

    # Merge all annotations
    annotations: list[str] = []
    for split in [train, val, test]:
        annotations += split["annotations"]

    lmdb_store = Ego4DHLMDB(args.path_to_output_lmdbs, frame_template=args.fname_format)

    # Define the dataset and dataloader
    dest = PyAVSTADataset(
        video_uid=args.video_uid,
        annotations=annotations,
        path_to_videos=args.path_to_videos,
        existing_keys=lmdb_store.get_existing_keys(),
        frame_height=args.frame_height,
    )
    dataloader = DataLoader(dest, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

    # Iterate over the dataloader
    for batch in tqdm(dataloader):
        frames = batch["ims"]
        keys = batch["keys"]
        for parent in np.unique([k.rsplit("_", 1)[0] for k in keys]):
            idx = np.where([k.startswith(parent + "_") for k in keys])[0]
            these_keys = [int(keys[i].rsplit("_", 1)[1]) for i in idx]
            these_frames = [frames[i] for i in idx]
            lmdb_store.put_batch(parent, these_keys, these_frames)


class Annotation(TypedDict):
    video_uid: str
    frame: int


class FrameData(TypedDict):
    video_id: list[npt.NDArray[np.int_]]
    frame_number: int
    frame: np.ndarray


class FrameChunk(TypedDict):
    frame_numbers: list[int]
    imgs: list[npt.NDArray[np.uint8]]


class LMDBChunk(TypedDict):
    ims: list[npt.NDArray[np.uint8]]
    keys: list[str]


class PyAVSTADataset(Dataset[LMDBChunk]):
    def __init__(
        self,
        video_uid: list[str] | None,
        annotations: list[Annotation],
        path_to_videos: Path,
        existing_keys: list[bytes],
        context_frames: int = 32,
        fps: int = 30,
        max_chunk_size: int = 32,
        frame_height: int = 320,
        retry: int = 10,
    ) -> None:
        """Initialize the dataset with annotations, video paths, and existing keys.

        Args:
            video_uid: List of unique video identifiers to filter the annotations.
            annotations: List of annotations for the dataset.
            path_to_videos: Path to the directory containing the video files.
            existing_keys: List of existing keys in the LMDBs to avoid duplicates.
            context_frames: Number of context frames to use for each sample.
            fps: Frames per second of the videos.
            max_chunk_size: Maximum number of frames to process in a single chunk.
            frame_height: Height of the video frames.
            retry: Number of times to retry loading a video in case of failure.
        """
        print(f"Sampling from {len(annotations)} annotations with a temporal context of {context_frames / fps} seconds")
        existing_frames: dict[str, list[int]] = defaultdict(list)
        for key in existing_keys:
            key_str = str(key.decode("utf-8"))
            video_id, frame_number = key_str.rsplit("_", 1)
            existing_frames[video_id].append(int(frame_number))

        self.path_to_videos = path_to_videos
        self.retry = retry
        self.frame_height = frame_height
        if video_uid is not None:
            annotations = [a for a in annotations if a["video_uid"] in video_uid]

        frames_per_video: dict[str, list[int]] = defaultdict(list)
        for annotation in annotations:
            video_id = annotation["video_uid"]
            last_frame = annotation["frame"]
            first_frame = np.max([0, last_frame - context_frames + 1])
            frame_numbers = np.arange(first_frame, last_frame + 1)
            frames_per_video[video_id].extend(frame_numbers)

        self.chunks: list[tuple[str, npt.NDArray[np.int_]]] = []

        total_frames = 0

        for video_id, frame_numbers in frames_per_video.items():
            frames = np.setdiff1d(np.sort(np.unique(frame_numbers)), existing_frames[video_id])

            if len(frames) > 0:
                # Break at non consecutive frames
                frame_chunks = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
                # Add each frame chunk to the list of chunks
                for chunk in frame_chunks:
                    if len(chunk) <= max_chunk_size:
                        self.chunks.append((video_id, chunk))
                        total_frames += len(chunk)
                    else:
                        for chunk in np.array_split(chunk, np.ceil(len(chunk) / max_chunk_size)):
                            self.chunks.append((video_id, chunk))
                            total_frames += len(chunk)

        total_frames += len(existing_keys)

        avg_bytes = 60000
        total_bytes = total_frames * avg_bytes
        total_gigabytes = total_bytes / 1024 / 1024 / 1024

        print(f"Sampled {len(self.chunks)} chunks / {total_frames} frames in total")
        print(f"Skipping {len(existing_keys)} existing keys")
        print(f"Estimated total size: {total_gigabytes:0.2f} GB")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> LMDBChunk:
        video_id, frame_numbers = self.chunks[idx]

        collected: dict[int, npt.NDArray[np.uint8]] = {}

        for _ in range(self.retry):
            missing = np.setdiff1d(frame_numbers, list(collected.keys()))
            if len(missing) == 0:
                break
            vr = PyAVVideoReader(self.path_to_videos / (video_id + ".mp4"), height=self.frame_height)
            ims = vr[missing]
            for frame_num, im in zip(missing, ims, strict=True):
                if im is not None:
                    collected[int(frame_num)] = im

        missing_frames = np.setdiff1d(frame_numbers, list(collected.keys()))
        if len(missing_frames) > 0:
            print(
                f"WARNING: could not read the following frames from {video_id}:",
                ", ".join([str(x) for x in missing_frames]),
            )

        result_ims = [collected[int(fn)] for fn in frame_numbers if int(fn) in collected]
        result_keys = [f"{video_id}_{fn}" for fn in frame_numbers if int(fn) in collected]
        return {"ims": result_ims, "keys": result_keys}


def collate_fn(batch: list[LMDBChunk]) -> LMDBChunk:
    """Collate function for the DataLoader to combine multiple samples into a batch.

    Args:
        batch: List of samples, where each sample is a dictionary containing 'ims' and 'keys'.

    Returns:
        A dictionary containing a list of frames under 'ims' and a list of corresponding keys under 'keys'.
    """
    frames = [sample["ims"] for sample in batch]
    keys = [sample["keys"] for sample in batch]
    frames = list(chain.from_iterable(frames))
    keys = list(chain.from_iterable(keys))

    return {"ims": frames, "keys": keys}


if __name__ == "__main__":
    main()
