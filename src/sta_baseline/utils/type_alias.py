from __future__ import annotations

from enum import Enum
from typing import TypedDict


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


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
