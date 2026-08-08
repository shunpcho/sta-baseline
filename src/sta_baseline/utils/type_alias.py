from __future__ import annotations

from enum import Enum
from typing import NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt
import torch


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ----------------------------------------------------------------------------------------------------------------------
# Type aliases for Short-Term Anticipation Annotations
# ----------------------------------------------------------------------------------------------------------------------


class FHOSTA(TypedDict):
    info: FHOSTAInfo
    annotations: list[FHOSTAAnnotation]


class FHOSTAInfo(TypedDict):
    video_metadata: FHOVideoMetadata
    year: int
    date_created: str  # format: YYYY-MM-DD


type FHOVideoMetadata = dict[str, FrameInfo]  # The keys are video ids.


class FrameInfo(TypedDict):
    frame_width: int
    frame_height: int
    fps: float


class FHOSTAAnnotation(TypedDict):
    uid: str
    main_uid: str
    video_uid: str
    frame: int
    clip_frame: int
    clip_id: int
    clip_uid: str
    action_start_sec: float
    action_end_sec: float
    action_start_frame: int
    action_end_frame: int
    action_clip_start_sec: float
    action_clip_end_sec: float
    action_clip_start_frame: int
    action_clip_end_frame: int
    interval_start_frame: int
    interval_end_frame: int
    interval_start_sec: float
    interval_end_sec: float
    clip_parent_start_sec: float
    clip_parent_end_sec: float
    clip_parent_start_frame: int
    clip_parent_end_frame: int
    objects: ObjectsAnnotation


class ObjectsAnnotation(TypedDict):
    box: list[float]
    verb_category_id: int
    noun_category_id: int
    time_to_contact: float


type ObjectDetectionAnnotation = dict[str, list[ObjectBoxes]]


class ObjectBoxes(TypedDict):
    box: list[float]
    score: float
    noun_category_id: int


# ----------------------------------------------------------------------------------------------------------------------
# Type aliases for Short-Term Anticipation Dataset, Data and Batch
# ----------------------------------------------------------------------------------------------------------------------


class ShortTermAnticipationData(NamedTuple):
    uid: str
    images: list[torch.Tensor]
    pred_boxes: npt.NDArray[np.float32]
    verb_labels: npt.NDArray[np.int32]
    ttc_targets: npt.NDArray[np.float32]
    extra_data: ExtraData


class ShortTermAnticipationBatch(NamedTuple):
    uids: list[str]
    images: torch.Tensor
    pred_boxes: list[torch.Tensor]
    verb_labels: list[torch.Tensor]
    ttc_targets: list[torch.Tensor]
    extras: ExtraDataBatch


class ExtraData(TypedDict):
    orig_pred_boxes: npt.NDArray[np.float32]
    pred_object_scores: npt.NDArray[np.float32]
    pred_object_labels: npt.NDArray[np.int32]
    gt_detections: Detections | None


class ExtraDataBatch(TypedDict):
    orig_pred_boxes: list[npt.NDArray[np.float32]]
    pred_object_scores: list[npt.NDArray[np.float32]]
    pred_object_labels: list[npt.NDArray[np.int32]]
    gt_detections: list[Detections | None]


class Detections(TypedDict):
    boxes: npt.NDArray[np.float32]
    nouns: npt.NDArray[np.int32]
    verbs: npt.NDArray[np.int32]
    ttcs: npt.NDArray[np.float32]


class STAModelOutput(NamedTuple):
    detections: list[Detections]
    raw_predictions: list[dict[str, torch.Tensor]]
