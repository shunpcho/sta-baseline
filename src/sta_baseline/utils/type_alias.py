from __future__ import annotations

from typing import TypedDict


class FHOSTAAnnotation(TypedDict):
    uid: str
    main_uid: str
    video_uid: str
    frame: int
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
    verb_category: int
    noun_category: int
    time_to_contact: float
