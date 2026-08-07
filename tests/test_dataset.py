from pathlib import Path
from typing import NamedTuple

import pytest
from fvcore.common.config import CfgNode

from sta_baseline.datasets.short_term_anticipation import Ego4dShortTermAnticipation
from sta_baseline.utils.type_alias import FHOSTAInfo, FrameInfo, ObjectBoxes, Split
from tests.conftest import (
    create_dummy_json_object_detections,
    create_dummy_lmdb,
    create_dummy_sta_train_json,
    DummyFHOSTA,
)


class DummyPaths(NamedTuple):
    lmdb_path: Path
    object_detections_path: Path
    sta_annotation_path: Path


dummy_video_id = "dummy162435"
frame_numbers = [7, 8, 9, 10]
movie_frame_size = (128, 96)
dummy_object_detection = {
    f"{dummy_video_id}_0000007": [ObjectBoxes(box=[10.1, 20.2, 30.3, 40.4], score=0.34867634, noun_category_id=71)],
    f"{dummy_video_id}_0000008": [ObjectBoxes(box=[11.1, 21.2, 31.3, 41.4], score=0.44867634, noun_category_id=81)],
    f"{dummy_video_id}_0000009": [ObjectBoxes(box=[12.1, 22.2, 32.3, 42.4], score=0.54867634, noun_category_id=91)],
    f"{dummy_video_id}_0000010": [ObjectBoxes(box=[13.1, 23.2, 33.3, 43.4], score=0.64867634, noun_category_id=101)],
}
dummy_sta_train_annotations = DummyFHOSTA(
    info=FHOSTAInfo(
        video_metadata={
            dummy_video_id: FrameInfo(
                frame_width=movie_frame_size[0],
                frame_height=movie_frame_size[1],
                fps=30.0,
            )
        },
        year=2024,
        date_created="2024-06-01",
    ),
    annotations=[
        {
            "uid": f"{dummy_video_id}_0000007",
            "main_uid": "dummy_main_1",
            "video_uid": dummy_video_id,
            "frame": 1136,
            "clip_id": 1,
            "clip_uid": dummy_video_id,
        },
        {
            "uid": f"{dummy_video_id}_0000008",
            "main_uid": "dummy_main_1",
            "video_uid": dummy_video_id,
            "frame": 1136,
            "clip_id": 1,
            "clip_uid": dummy_video_id,
        },
        {
            "uid": f"{dummy_video_id}_0000009",
            "main_uid": "dummy_main_1",
            "video_uid": dummy_video_id,
            "frame": 1136,
            "clip_id": 1,
            "clip_uid": dummy_video_id,
        },
        {
            "uid": f"{dummy_video_id}_0000010",
            "main_uid": "dummy_main_1",
            "video_uid": dummy_video_id,
            "frame": 1136,
            "clip_id": 1,
            "clip_uid": dummy_video_id,
        },
    ],
)


@pytest.fixture
def dummy_lmdb_and_detections(tmp_path: Path) -> DummyPaths:
    """Create a dummy LMDB and object detections for testing."""
    lmdb_path = create_dummy_lmdb(tmp_path, dummy_video_id, frame_numbers=frame_numbers, image_size=movie_frame_size)
    object_detections_path = create_dummy_json_object_detections(tmp_path, dummy_object_detection)
    sta_annotation_path = create_dummy_sta_train_json(tmp_path, sta_train_annotations=dummy_sta_train_annotations)
    return DummyPaths(
        lmdb_path=lmdb_path,
        object_detections_path=object_detections_path,
        sta_annotation_path=sta_annotation_path,
    )


def test_sta_dataset(dummy_lmdb_and_detections: DummyPaths) -> None:
    """Test that Ego4dShortTermAnticipation can be instantiated with dummy LMDB and object detections."""
    cfg = CfgNode()
    cfg.DATA = CfgNode()
    cfg.DATA.SAMPLING_RATE = 8
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.MEAN = [0.45, 0.45, 0.45]
    cfg.DATA.STD = [0.225, 0.225, 0.225]
    cfg.DATA.TARGET_FPS = 30
    cfg.DATA.RANDOM_FLIP = True
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
    cfg.DATA.TEST_CROP_SIZE = 256

    cfg.MODEL = CfgNode()
    cfg.MODEL.NUM_CLASSES = [400]

    cfg.EGO4D_STA = CfgNode()
    cfg.EGO4D_STA.BGR = False
    cfg.EGO4D_STA.TRAIN_USE_COLOR_AUGMENTATION = False
    cfg.EGO4D_STA.TRAIN_PCA_JITTER_ONLY = False
    cfg.EGO4D_STA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]
    cfg.EGO4D_STA.TRAIN_PCA_EIGVEC = [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]
    cfg.EGO4D_STA.TEST_FORCE_FLIP = False
    cfg.EGO4D_STA.RGB_LMDB_DIR = str(dummy_lmdb_and_detections.lmdb_path)
    cfg.EGO4D_STA.VIDEO_LOAD_BACKEND = "lmdb"
    cfg.EGO4D_STA.OBJ_DETECTIONS = str(dummy_lmdb_and_detections.object_detections_path)
    cfg.EGO4D_STA.ANNOTATION_DIR = str(dummy_lmdb_and_detections.sta_annotation_path.parent)
    cfg.EGO4D_STA.TRAIN_LISTS = ["fho_sta_train.json"]
    cfg.EGO4D_STA.VAL_LISTS = ["fho_sta_val.json"]
    cfg.EGO4D_STA.TEST_LISTS = ["fho_sta_test_unannotated.json"]

    dataset = Ego4dShortTermAnticipation(cfg, split=Split.TRAIN)

    assert len(dataset) == 4
