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
from sta_baseline.utils import logging
from sta_baseline.utils.type_alias import FHOSTAAnnotation

logger = logging.get_logger(__name__)


def main() -> None:
    """Main function to parse command-line arguments and create LMDBs."""
    logging.setup_logging()

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
    fho_sta_annotations: list[FHOSTAAnnotation] = []
    for split in [train, val, test]:
        fho_sta_annotations += split["annotations"]

    # Filter annotations
    annotations: list[LMDBAnnotation] = [
        LMDBAnnotation(
            video_uid=annotation.get("video_uid", ""),
            frame=annotation.get("frame", 0),
            clip_uid=annotation.get("clip_uid", ""),
            clip_frame=annotation.get("clip_frame", 0),
        )
        for annotation in fho_sta_annotations
    ]

    non_empty_video_uid_count = sum(1 for annotation in annotations if annotation["video_uid"])
    non_empty_clip_uid_count = sum(1 for annotation in annotations if annotation["clip_uid"])
    unique_video_uids = {annotation["video_uid"] for annotation in annotations if annotation["video_uid"]}
    unique_clip_uids = {annotation["clip_uid"] for annotation in annotations if annotation["clip_uid"]}
    logger.info(
        "Annotation stats: "
        f"annotations={len(annotations)}, "
        f"video_uid_non_empty={non_empty_video_uid_count}, "
        f"clip_uid_non_empty={non_empty_clip_uid_count}, "
        f"video_uid={len(unique_video_uids)}, "
        f"clip_uid={len(unique_clip_uids)}"
    )

    # Which 'video_uid' or 'clip_uid' to sample from.
    flag = "clip_uid" if args.path_to_videos.name == "clips" else "video_uid"

    lmdb_store = Ego4DHLMDB(args.path_to_output_lmdbs, frame_template=args.fname_format)

    # Define the dataset and dataloader
    dest = PyAVSTADataset(
        flag=flag,
        video_uid=args.video_uid,
        annotations=annotations,
        path_to_videos=args.path_to_videos,
        existing_keys=lmdb_store.get_existing_keys(),
        frame_height=args.frame_height,
        context_frames=args.context_frames,
        fname_format=args.fname_format,
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


class LMDBAnnotation(TypedDict):
    video_uid: str
    frame: int
    clip_uid: str
    clip_frame: int


def _get_annotation_video_id(annotation: LMDBAnnotation, flag: str) -> str:
    if flag == "clip_uid" and "clip_uid" in annotation:
        return annotation["clip_uid"]
    if flag == "video_uid" and "video_uid" in annotation:
        return annotation["video_uid"]
    raise KeyError(f"Annotation must contain the '{flag}' key")


def _get_annotation_frame_number(annotation: Annotation, flag: str) -> int:
    if flag == "clip_uid" and "clip_frame" in annotation:
        return annotation["clip_frame"]
    if flag == "video_uid" and "frame" in annotation:
        return annotation["frame"]
    raise KeyError(f"Annotation must contain the frame number for '{flag}'")


class LMDBChunk(TypedDict):
    ims: list[npt.NDArray[np.uint8]]
    keys: list[str]


class PyAVSTADataset(Dataset[LMDBChunk]):
    def __init__(
        self,
        flag: str,
        video_uid: list[str] | None,
        annotations: list[LMDBAnnotation],
        path_to_videos: Path,
        existing_keys: list[bytes],
        context_frames: int = 32,
        fps: int = 30,
        max_chunk_size: int = 32,
        frame_height: int = 320,
        fname_format: str = "{video_id:s}_{frame_number:07d}",
        retry: int = 10,
    ) -> None:
        """Initialize the dataset with annotations, video paths, and existing keys.

        Args:
            flag: Either 'clip_uid' or 'video_uid' to determine which identifier to use.
            video_uid: List of unique video identifiers to filter the annotations.
            annotations: List of annotations for the dataset.
            path_to_videos: Path to the directory containing the video files.
            existing_keys: List of existing keys in the LMDBs to avoid duplicates.
            context_frames: Number of context frames to use for each sample.
            fps: Frames per second of the videos.
            max_chunk_size: Maximum number of frames to process in a single chunk.
            frame_height: Height of the video frames.
            fname_format: Format string for generating frame keys.
            retry: Number of times to retry loading a video in case of failure.
        """
        logger.info("Video UID filter: %s", flag)
        logger.info(
            "Sampling from %d annotations with a temporal context of %.3f seconds",
            len(annotations),
            context_frames / fps,
        )

        existing_frames: dict[str, list[int]] = defaultdict(list)
        for key in existing_keys:
            key_str = str(key.decode("utf-8"))
            video_id, frame_number = key_str.rsplit("_", 1)
            existing_frames[video_id].append(int(frame_number))

        self.path_to_videos = path_to_videos
        self.retry = retry
        self.frame_height = frame_height
        self.fname_format = fname_format
        if video_uid is not None:
            annotations = [a for a in annotations if _get_annotation_video_id(a, flag) in video_uid]

        frames_per_video: dict[str, list[int]] = defaultdict(list)
        for annotation in annotations:
            video_id = _get_annotation_video_id(annotation, flag)
            last_frame = _get_annotation_frame_number(annotation, flag)
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
                        for sub_chunk in np.array_split(chunk, int(np.ceil(len(chunk) / max_chunk_size))):
                            self.chunks.append((video_id, sub_chunk))
                            total_frames += len(sub_chunk)

        total_frames += len(existing_keys)

        avg_bytes = 60000
        total_bytes = total_frames * avg_bytes
        total_gigabytes = total_bytes / 1024 / 1024 / 1024

        logger.info("Sampled %d chunks / %d frames in total", len(self.chunks), total_frames)
        logger.info("Skipping %d existing keys", len(existing_keys))
        logger.info("Estimated total size: %.2f GB", total_gigabytes)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> LMDBChunk:
        video_id, frame_numbers = self.chunks[idx]

        if len(frame_numbers) == 0:
            return LMDBChunk(ims=[], keys=[])

        video_path = self.path_to_videos / (video_id + ".mp4")

        frames: dict[int, npt.NDArray[np.uint8]] = {}

        for _ in range(self.retry):
            remaining_frame_numbers = np.setdiff1d(frame_numbers, list(frames.keys()))
            if len(remaining_frame_numbers) == 0:
                ordered_frame_numbers = sorted(frames)
                return LMDBChunk(
                    ims=[frames[frame_number] for frame_number in ordered_frame_numbers],
                    keys=[
                        self.fname_format.format(video_id=video_id, frame_number=frame_number)
                        for frame_number in ordered_frame_numbers
                    ],
                )

            try:
                vr = PyAVVideoReader(str(video_path), height=self.frame_height)
                ims = vr[remaining_frame_numbers]
            except FileNotFoundError:
                logger.warning("video not found, skipping %s: %s", video_id, video_path)
                return LMDBChunk(ims=[], keys=[])

            added_frames = 0
            for frame_number, img in zip(remaining_frame_numbers, ims, strict=True):
                if img is not None:
                    frames[frame_number] = img
                    added_frames += 1

            if added_frames == len(remaining_frame_numbers):
                ordered_frame_numbers = sorted(frames)
                return LMDBChunk(
                    ims=[frames[frame_number] for frame_number in ordered_frame_numbers],
                    keys=[
                        self.fname_format.format(video_id=video_id, frame_number=frame_number)
                        for frame_number in ordered_frame_numbers
                    ],
                )

            missing_frames = np.setdiff1d(remaining_frame_numbers, list(frames.keys()))

            if len(missing_frames) > 0:
                logger.warning(
                    "could not read the following frames from %s: %s",
                    video_id,
                    ", ".join([str(x) for x in missing_frames]),
                )

        ordered_frame_numbers = sorted(frames)
        return LMDBChunk(
            ims=[frames[frame_number] for frame_number in ordered_frame_numbers],
            keys=[
                self.fname_format.format(video_id=video_id, frame_number=frame_number)
                for frame_number in ordered_frame_numbers
            ],
        )


def collate_fn(batch: list[LMDBChunk]) -> LMDBChunk:
    """Collate function for the DataLoader to combine multiple samples into a batch.

    Args:
        batch: List of samples, where each sample is a dictionary containing 'ims' and 'keys'.

    Returns:
        A dictionary containing a list of frames under 'ims' and a list of corresponding keys under 'keys'.
    """
    batch = [sample for sample in batch if sample["ims"] or sample["keys"]]
    frames = [sample["ims"] for sample in batch]
    keys = [sample["keys"] for sample in batch]
    frames = list(chain.from_iterable(frames))
    keys = list(chain.from_iterable(keys))

    return {"ims": frames, "keys": keys}


if __name__ == "__main__":
    main()
