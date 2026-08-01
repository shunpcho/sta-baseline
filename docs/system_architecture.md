# System Architecture of STA Baseline

This document describes the system architecture of a baseline for Short-Term Anticipation (STA).
It is based on the EGO4D [Short-Term Anticipation task specification](https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md).

## Table of Contents

- [Tasks](#tasks)
- [System Architecture](#system-architecture)
  - [1. Extract 32 Frames](#1-extract-32-frames)
  - [2. Object Detection](#2-object-detection)
  - [3. Predict Verb Labels and Estimate Time to Contact](#3-predict-verb-labels-and-estimate-time-to-contact)
- [Data](#data)
  - [STA Annotations](#sta-annotations)
  - [Hand Boxes](#hand-boxes)
  - [FHO Annotation Information](#fho-annotation-information)

## Tasks

The task predicts a camera wearer's future interactions from first-person video. Predictions are made at specified timestamps rather than by exhaustively searching the entire video.

- The bounding box of the object the camera wearer will next interact with.
- The noun category of that object.
- The verb describing the anticipated interaction.
- The time until contact (TTC).

The task is evaluated using Noun Top-5 mAP, Noun+Verb Top-5 mAP, Noun+TTC Top-5 mAP, and Overall Top-5 mAP.

## System Architecture

The system consists of three steps:

1. Extract 32 frames.
2. Detect objects.
3. Predict verb labels and estimate time to contact (TTC).

### 1. Extract 32 Frames

Extract the 32 frames preceding each annotated timestamp. This requires preprocessing the video frames and annotations into LMDB.

### 2. Object Detection

Apply image-based object detection to the extracted frames.

### 3. Predict Verb Labels and Estimate Time to Contact

Use [SlowFast](https://github.com/facebookresearch/SlowFast) to predict action labels and time to contact.

## Data

The dataset provides video frames and annotation data. Each annotation represents one anticipated interaction at a specific timestamp.

### STA Annotations

STA annotation files contain metadata and annotations.

#### `fho_sta_<split>.json`

```json
{
  "info": {
    "description": "...",
    "version": "2.0",
    "split": "train",
    "include_annotations": true,
    "video_metadata": { ... }
  },
  "annotations": [ ... ]
}
```

Each entry in `"annotations"` describes an anticipation sample:

```json
{
  "uid": "unique_id",
  "video_uid": "video_uid",
  "frame": <frame_number>,
  "clip_id": "...",
  "clip_uid": "...",
  "clip_frame": <frame_index_in_clip>,
  "objects": [ ... ]
}
```

- Each object in `"objects"` contains the following fields:
  - The `verb` has 98 classes.
  - The `noun` has 301 classes.

```json
{
  "box": [x1, y1, x2, y2],
  "verb_category_id": <int>,
  "noun_category_id": <int>,
  "time_to_contact": <float>
}
```

`"video_metadata"` contains each video's full resolution, for example, 1920 x 1080 (1080p) or 1280 x 720 (720p).

```json
{
  "info": {
    "video_metadata": {
      "<video_uid>": {
        "frame_width": <int>,
        "frame_height": <int>,
        "fps": <float>
      }
    }
  }
}
```

#### Hand Boxes

Hand-box annotations provide the bounding-box coordinates for the left and right hands in each frame.

`fho_hands_<split>.json`

```json
{
  "annotations": [
    {
      "clip_uid": "xxxx",
      "video_uid": "abcd",
      "frames": {
        "1234": {
          "left_hand": [x1, y1, x2, y2],
          "right_hand": [x1, y1, x2, y2]
        },
        "1235": {
          "left_hand": [...],
          "right_hand": [...]
        },
        ...
      }
    },
    ...
  ]
}
```

### FHO annotation information

`fho_main.json` contains the master annotations for forecasting.

```json
{
  "version": "2.0",
  "date": "yymmdd",
  "description": "FHO Master Annotation",
  "metadata": "s3://ego4d-consortium-sharing/public/v2/ego4d.json",
  "videos": [
    {
      "annotated_intervals": [
        {
          "clip_id": "451",
          "clip_uid": "a102c79b-405b-4a13-b21e-ab7dc6135b22",
          "start_sec": 0.0,
          "end_sec": 207.63,
          "clip_parent_start_sec": 0.0,
          "clip_parent_end_sec": 207.633,
          "narrated_actions": [],
          "start_frame": 0,
          "end_frame": 6229,
          "clip_parent_start_frame": 0,
          "clip_parent_end_frame": 6229,
          "redacted": true
        }
      ]
    }
  ]
}
```

`fho_sta_train.json`

```json
{
  "info": {
    "description": "Ego4D Short-Term Object Interaction Anticipation Dataset",
    "version": "2.0",
    "split": "train",
    "include_annotations": true,
    "video_metadata": {
      "26202090-684d-4be8-b3cc-de04da827e91": {
        "frame_width": 1440,
        "frame_height": 1080,
        "fps": 30.0
      },
      "d8c894ab-7b08-4983-9e80-fdb5d6ee0202": {
        "frame_width": 1440,
        "frame_height": 1080,
        "fps": 30.0
      },
      "cde41c4f-50d1-4910-9f2a-4c7b6987df92": {
        "frame_width": 1920,
        "frame_height": 1440,
        "fps": 30.0
      }
    }
  }
}
```
