import io
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import decord
import imutils
import lmdb
import numpy as np
import numpy.typing as npt
import torch
from cv2 import imdecode, imencode, IMREAD_COLOR
from decord import VideoReader
from fvcore.common.config import CfgNode
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset

from sta_baseline.datasets import cv2_transform
from sta_baseline.datasets.build import DATASET_REGISTRY
from sta_baseline.evaluation.sta_evaluate import compute_iou
from sta_baseline.lib.pytorchvideo.transform_functional import uniform_temporal_subsample
from sta_baseline.utils import datasets_utils, logging, transform
from sta_baseline.utils.type_alias import Split

type FrameSequence = int | list[int] | tuple[int, ...] | npt.NDArray[np.int_]

logger = logging.get_logger(__name__)
decord.bridge.set_bridge("torch")


def _get_frames(
    frame_list: list[int],
    container: av.container.Container,
    include_audio: bool = False,
    audio_buffer_frames: int = 0,
) -> list[av.frame.Frame | None]:
    if len(frame_list) == 0:
        return []

    if len(container.streams.video) == 0:
        raise ValueError(f"No video streams found in {container.name}")
    if len(container.streams.video) > 1:
        raise ValueError(f"Multiple video streams not supported in {container.name}")

    video_stream = container.streams.video[0]
    video_start = video_stream.start_time or 0
    video_base = video_stream.time_base
    fps = video_stream.average_rate
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    audio_buffer_pts = frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff) if include_audio else 0

    pts_to_idx: dict[int, int] = {
        frame_index_to_pts(frame_idx, video_start, video_pt_diff): list_idx
        for list_idx, frame_idx in enumerate(frame_list)
    }

    first_pts = frame_index_to_pts(min(frame_list), video_start, video_pt_diff)
    container.seek(first_pts - audio_buffer_pts, stream=video_stream)

    result: list[av.frame.Frame | None] = [None] * len(frame_list)
    remaining = set(pts_to_idx.keys())

    for frame in container.decode(video=0):
        if frame.pts in pts_to_idx:
            result[pts_to_idx[frame.pts]] = frame
            remaining.discard(frame.pts)
        if not remaining:
            break

    return result


def pts_difference_per_frame(fps: Fraction, time_base: Fraction) -> int:
    return round(1 / fps / time_base)


def frame_index_to_pts(frame: int, start_pts: int, diff_per_frame: int) -> int:
    return start_pts + frame * diff_per_frame


class PyAVVideoReader:
    """To read frames from a video file using PyAV."""

    def __init__(
        self,
        path_to_video: str | Path,
        include_audio: bool = False,
        audio_buffer_frames: int = 0,
        height: int | None = None,
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

    def __getitem__(self, frame_list: FrameSequence) -> list[npt.NDArray[np.int_] | None]:
        """Get frames from the video based on the provided frame list.

        Args:
            frame_list: List of frame indices to retrieve.

        Returns:
            List of frames as numpy arrays. If a frame is not available, None is returned.
        """
        if isinstance(frame_list, (int, np.integer)):
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
            return imdecode(file_bytes, IMREAD_COLOR) if file_bytes is not None else None

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

    def get_existing_keys(self) -> list[bytes]:
        """Get a list of existing keys in the LMDB.

        Returns:
            List of existing keys as bytes.
        """
        existing_keys: list[bytes] = []
        for parent in self.path_to_root.iterdir():
            if parent.is_dir():
                with self._get_parent(parent.name) as env, env.begin(write=False) as txn:
                    existing_keys += list(txn.cursor().iternext(values=False))
        return existing_keys


@DATASET_REGISTRY.register()
class Ego4dShortTermAnticipation(Dataset):
    """Ego4d Short Term Anticipation Dataset."""

    def __init__(self, cfg: CfgNode, split: Split) -> None:
        # Only support train and val mode.

        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.EGO4D_STA.BGR
        self._fps = cfg.DATA.TARGET_FPS
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == Split.TRAIN:
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.EGO4D_STA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.EGO4D_STA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.EGO4D_STA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.EGO4D_STA.TRAIN_PCA_EIGVEC
        else:  # self._split == Split.VAL
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.EGO4D_STA.TEST_FORCE_FLIP

        if self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == "lmdb":
            self._hlmdb = Ego4DHLMDB(self.cfg.EGO4D_STA.RGB_LMDB_DIR, readonly=True, lock=False)

        with Path(cfg.EGO4D_STA.OBJ_DETECTIONS).open(encoding="utf-8") as f:
            self._obj_detections = json.load(f)

        self._load_data(cfg)

    def _load_lists(self, path_list: list[str]) -> dict:
        def extend_dict(input_dict: dict[str, Any], output_dict: dict[str, Any]) -> dict[str, Any]:
            return output_dict.update(input_dict) or output_dict

        res: dict[str, Any] = {"videos": {}, "annotations": []}
        for file in path_list:
            annotation_file_path = Path(self.cfg.EGO4D_STA.ANNOTATION_DIR, file)
            with annotation_file_path.open(encoding="utf-8") as f:
                j = json.load(f)
            res["videos"] = extend_dict(j["info"]["video_metadata"], res["videos"])
            res["annotations"] += j["annotations"]

        return res

    def _load_data(self, cfg: CfgNode) -> None:
        """Load frame paths and annotations from files.

        Args:
            cfg (CfgNode): config
        """
        if self._split == Split.TRAIN:
            self._annotations = self._load_lists(cfg.EGO4D_STA.TRAIN_LISTS)
        elif self._split == Split.VAL:
            self._annotations = self._load_lists(cfg.EGO4D_STA.VAL_LISTS)
        else:  # self._split == Split.TEST
            self._annotations = self._load_lists(cfg.EGO4D_STA.TEST_LISTS)

        annotations = self._annotations["annotations"]
        total_annotations = len(annotations)
        videos = self._annotations["videos"]
        valid_annotations: list[dict[str, Any]] = []
        skipped = 0
        for ann in annotations:
            video_id = ann.get("clip_uid")
            if isinstance(video_id, str) and video_id in videos:
                valid_annotations.append(ann)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning("Skipped %d annotations with missing or unknown clip_uid.", skipped)
        self._annotations["annotations"] = valid_annotations
        split_name = getattr(self._split, "value", str(self._split))
        logger.info(
            "Loaded %d annotations for split=%s (raw=%d, skipped=%d).",
            len(valid_annotations),
            split_name,
            total_annotations,
            skipped,
        )

    def __len__(self) -> int:
        return len(self._annotations["annotations"])

    def _images_and_boxes_preprocessing_cv2(
        self, imgs: torch.Tensor, boxes: npt.NDArray[np.uint8]
    ) -> tuple[torch.Tensor, npt.NDArray[np.float32]]:
        """Preprocessing for the input images and corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == Split.TRAIN:  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(imgs, self._crop_size, order="HWC", boxes=boxes)

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(0.5, imgs, order="HWC", boxes=boxes)
        elif self._split == Split.VAL:
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(self._crop_size, imgs, 1, boxes=boxes)

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(1, imgs, order="HWC", boxes=boxes)
        elif self._split == Split.TEST:
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(1, imgs, order="HWC", boxes=boxes)
        else:
            raise NotImplementedError(f"Unsupported split mode {self._split}")

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.hwc_to_chw(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == Split.TRAIN and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4)

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(boxes[0], imgs[0].shape[1], imgs[0].shape[2])
        return imgs, boxes

    def _images_and_boxes_preprocessing(
        self, imgs: torch.Tensor, boxes: npt.NDArray[np.float32]
    ) -> tuple[torch.Tensor, npt.NDArray[np.float32]]:
        """Preprocessing for the input images and corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs /= 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == Split.TRAIN:
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(imgs, self._crop_size, boxes=boxes)

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == Split.VAL:
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs, min_size=self._crop_size, max_size=self._crop_size, boxes=boxes
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(imgs, size=self._crop_size, spatial_idx=1, boxes=boxes)

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == Split.TEST:
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs, min_size=self._crop_size, max_size=self._crop_size, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(f"{self._split} split not supported yet!")

        # Do color augmentation (after divided by 255.0).
        if self._split == Split.TRAIN and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4)

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if self._use_bgr:
            # Convert image format from RGB to BGR.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(boxes, self._crop_size, self._crop_size)

        return imgs, boxes

    def _load_frames_decord(self, video_filename: str, frame_number: int) -> npt.NDArray[np.uint8]:
        assert frame_number > 0

        vr = VideoReader(video_filename, height=320, width=568)

        frames = (
            frame_number
            - np.arange(
                self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE,
                step=self.cfg.DATA.SAMPLING_RATE,
            )[::-1]
        )
        frames[frames < 1] = 1

        frames = frames.astype(int)

        video_data = vr.get_batch(frames).permute(3, 0, 1, 2)
        return video_data

    def _load_frames_pyav(self, video_filename: str, frame_number: int) -> npt.NDArray[np.uint8]:
        assert frame_number > 0

        vr = PyAVVideoReader(video_filename, height=320)

        frames = (
            frame_number
            - np.arange(
                self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE,
                step=self.cfg.DATA.SAMPLING_RATE,
            )[::-1]
        )
        frames[frames < 1] = 1

        frames = frames.astype(int)

        imgs = vr[frames]

        return imgs

    def _load_frames_pytorch_video(self, video_filename: str, frame_number: int, fps: float) -> torch.Tensor:
        clip_duration = (self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE - 1) / fps
        clip_end_sec = frame_number / fps
        clip_start_sec = clip_end_sec - clip_duration

        # truncate if negative timestamp
        clip_start_sec = np.max(clip_start_sec, 0)

        video = EncodedVideo.from_path(video_filename, decode_audio=False)
        video_data = video.get_clip(clip_start_sec, clip_end_sec)["video"]
        video_data = uniform_temporal_subsample(video_data, self.cfg.DATA.NUM_FRAMES)
        # video_data = short_side_scale(video_data, )
        return video_data

    def _retry_load_images_lmdb(
        self, video_id: str, frames: list[int], retry: int = 10, backend: str = "pytorch"
    ) -> list[npt.NDArray[np.uint8] | None] | torch.Tensor:
        """This function is to load images with support of retrying for failed load.

        Args:
            video_id (str): ID of the video.
            frames (list[int]): list of frame numbers to be loaded.
            retry (int, optional): maximum time of loading retrying. Defaults to 10.
            backend (str): `pytorch` or `cv2`.

        Returns:
            imgs (list): list of loaded images.

        Raises:
            RuntimeError: if failed to load images after `retry` times.
        """
        for i in range(retry):
            imgs: list[npt.NDArray[np.uint8] | None] = []
            imgs = self._hlmdb.get_batch(video_id, frames)

            if all(img is not None for img in imgs):
                if backend == "pytorch":
                    torch_imgs = torch.as_tensor(np.stack(imgs))
                return torch_imgs
            else:
                logger.warning("Reading failed. Will retry.")
                time.sleep(1.0)
            if i == retry - 1:
                raise RuntimeError(f"Failed to load images from {video_id} after {retry} retries.")

        return imgs

    def _sample_frames(self, frame: int) -> list[int]:
        frames = (
            frame
            - np.arange(
                self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE,
                step=self.cfg.DATA.SAMPLING_RATE,
            )[::-1]
        )
        frames[frames < 0] = 0

        frames = frames.astype(int)

        return frames

    def _load_annotations(self, idx: int) -> tuple[Any, ...]:
        # get the idx-th annotation
        ann = self._annotations["annotations"][idx]
        uid = ann["uid"]

        # get video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels and gt_ttc_targets
        video_id = ann.get("clip_uid")
        if not isinstance(video_id, str) or video_id not in self._annotations["videos"]:
            raise KeyError(f"Invalid clip_uid in annotation: uid={uid}")
        frame_number = ann["frame"]

        if "objects" in ann:
            gt_boxes = np.vstack([x["box"] for x in ann["objects"]])
            gt_noun_labels = np.array([x["noun_category_id"] for x in ann["objects"]])
            gt_verb_labels = np.array([x["verb_category_id"] for x in ann["objects"]])
            gt_ttc_targets = np.array([x["time_to_contact"] for x in ann["objects"]])
        else:
            gt_boxes = gt_noun_labels = gt_verb_labels = gt_ttc_targets = None

        frame_width, frame_height = (
            self._annotations["videos"][video_id]["frame_width"],
            self._annotations["videos"][video_id]["frame_height"],
        )

        fps = self._annotations["videos"][video_id]["fps"]

        return (
            uid,
            video_id,
            frame_width,
            frame_height,
            frame_number,
            fps,
            gt_boxes,
            gt_noun_labels,
            gt_verb_labels,
            gt_ttc_targets,
        )

    def _load_detections(self, uid: str) -> tuple[Any, Any, Any]:
        # get the object detections for the current example
        object_detections = self._obj_detections[uid]

        if len(object_detections) > 0:
            pred_boxes = np.vstack([x["box"] for x in object_detections])
            pred_scores = np.array([x["score"] for x in object_detections])
            pred_object_labels = np.array([x["noun_category_id"] for x in object_detections])

            # exclude detections below the theshold
            detected = pred_scores >= self.cfg.EGO4D_STA.DETECTION_SCORE_THRESH

            pred_boxes = pred_boxes[detected]
            pred_object_labels = pred_object_labels[detected]
            pred_scores = pred_scores[detected]
        else:
            pred_boxes = np.zeros((0, 4))
            pred_scores = pred_object_labels = np.array([])

        return pred_boxes, pred_object_labels, pred_scores

    def _load_frames(self, video_id: str, frame_number: int, fps: float) -> torch.Tensor:
        video_path = Path(self.cfg.EGO4D_STA.VIDEOS_DIR, video_id + ".mp4")

        if self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == "pytorchvideo":
            frames = self._load_frames_pytorch_video(str(video_path), frame_number, fps)
        elif self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == "decord":
            frames = self._load_frames_decord(str(video_path), frame_number, fps)
        elif self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == "pyav":
            frames = self._load_frames_pyav(str(video_path), frame_number, fps)
        elif self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == "lmdb":
            # sample the list of frames in the clip
            # key_list = self._sample_frame_keys(video_id, frame_number)
            frames_list = self._sample_frames(frame_number)
            # # retrieve frames
            frames = self._retry_load_images_lmdb(video_id, frames_list, backend="cv2")
        return frames

    def _preprocess_frames_and_boxes(
        self, frames: torch.Tensor, boxes: npt.NDArray[np.uint8]
    ) -> tuple[torch.Tensor, np.NDArray[np.float32]]:
        if self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND in {"pytorchvideo", "decord"}:
            video_tensor = frames.permute(1, 0, 2, 3)

            video_tensor, boxes = self._images_and_boxes_preprocessing(video_tensor, boxes=boxes)

            # T C H W -> C T H W.
            video_tensor = video_tensor.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            video_tensor, boxes = self._images_and_boxes_preprocessing_cv2(frames, boxes=boxes)
        return video_tensor, boxes

    def __getitem__(self, idx: int) -> tuple[Any, ...]:  # noqa: PLR0914
        """Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.

        Returns:
            uid: the unique id of the annotation
            imgs: the frames sampled from the video
            pred_boxes: the list of boxes detected in the current frame. These are in the resolution of the input
                        example.
            verb_label: the verb label associated to the current frame
            ttc_target: the ttc target
            extra_data: a dictionary containing extra data fields:
                'orig_pred_boxes': boxes at the original resolution
                'pred_object_scores': associated prediction scores
                'pred_object_labels': associated predicted object labels
                'gt_detections': dictionary containing the ground truth predictions for the current frame
        """
        (
            uid,
            video_id,
            frame_width,
            frame_height,
            frame_number,
            fps,
            gt_boxes,
            gt_noun_labels,
            gt_verb_labels,
            gt_ttc_targets,
        ) = self._load_annotations(idx)
        pred_boxes, pred_object_labels, pred_scores = self._load_detections(uid)

        frames = self._load_frames(video_id, frame_number, fps)

        orig_pred_boxes = pred_boxes.copy()
        nn = np.array([frame_width, frame_height] * 2).reshape(1, -1)
        pred_boxes /= nn

        if gt_boxes is None:  # unlabeled example
            video_tensor, pred_boxes = self._preprocess_frames_and_boxes(frames, pred_boxes)
            imgs = datasets_utils.pack_pathway_output(self.cfg, video_tensor)

            extra_data = {
                "orig_pred_boxes": orig_pred_boxes,
                "pred_object_scores": pred_scores,
                "pred_object_labels": pred_object_labels,
            }

            return uid, imgs, pred_boxes, np.array([]), np.array([]), extra_data
        else:
            orig_gt_boxes = gt_boxes.copy()
            gt_boxes /= nn

            # put all boxes together
            all_boxes = np.vstack([gt_boxes, pred_boxes])

            video_tensor, all_boxes = self._preprocess_frames_and_boxes(frames, all_boxes)

            # separate ground truth from predicted boxes after pre-processing
            gt_boxes = all_boxes[: len(gt_boxes)]
            pred_boxes = all_boxes[len(gt_boxes) :]

            if self._split == "train" and self.cfg.EGO4D_STA.PROPOSAL_APPEND_GT:
                pred_boxes = np.concatenate([pred_boxes, gt_boxes])
                orig_pred_boxes = np.concatenate([orig_pred_boxes, orig_gt_boxes])
                pred_object_labels = np.concatenate([pred_object_labels, gt_noun_labels])
                pred_scores = np.concatenate([pred_scores, np.ones_like(gt_noun_labels)])

            # match predicted boxes to ground truth
            # compute IOU values
            ious = compute_iou(pred_boxes, gt_boxes)

            # get the indexes of the largest IOU - these are the matches
            matches = ious.argmax(-1)  # index of the matched gt_box for each pred_box

            # get the largest IOU for each predicted box
            ious = ious.max(-1)

            next_active_labels = ious >= self.cfg.EGO4D_STA.NAO_IOU_THRESH

            gt_detections = {
                "boxes": orig_gt_boxes,
                "nouns": gt_noun_labels,
                "verbs": gt_verb_labels,
                "ttcs": gt_ttc_targets,
            }

            imgs = datasets_utils.pack_pathway_output(self.cfg, video_tensor)

            # copy the verb labels of the matched boxes
            verb_labels = gt_verb_labels[matches]

            # set verb label to ignore index for not next active objects
            verb_labels[~next_active_labels] = -100

            # copy ttc targets of the matched boxes
            ttc_targets = gt_ttc_targets[matches]

            # set ttc targets related to non next-active objects to NaN
            ttc_targets[~next_active_labels] = np.nan

            extra_data = {
                "orig_pred_boxes": orig_pred_boxes,
                "pred_object_scores": pred_scores,
                "pred_object_labels": pred_object_labels,
                "gt_detections": gt_detections,
            }

            return (uid, imgs, pred_boxes, verb_labels, ttc_targets, extra_data)
