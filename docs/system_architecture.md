# System Architecture of STA Baseline

The system Arcitecture for building a baseline to solve Sort Term Anticipation (STA).
This is replica of Ego4D [forcasting/SHORT_TERM_ANTICIPATION.md](https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md)

## Tasks

一人称の動画から人が何をしようとしているかを予測する。動画中の全探索ではなく、特定のタイムスタンプごとに予測を行う。

- (主観者が次に作用する)オブジェクトのbbox
- (主観者が次に作用する)オブジェクトのカテゴリ
- 次にどのような行動をとるか
- オブジェクトと作用が開始するまでの予測時間

評価は「Noun Top-5 mAP」「Noun+Verb Top-5 mAP」「Noun+TTC Top-5 mAP」「Overall Top-5 mAP」で計算される。

## System Architecture

This system has three steps to solve the tasks.

1. Extract 32 frames
2. Object Detection
3. Predict verb labels and estimate time to contact (TTC).

### 1. Extract 32 frames

動画のAnnotationから前の32フレームを抽出する。（事前にフレーム抽出とアノテーションの作り替えが必要。）

### 2. Object Detection

普通にある画像物体検出

### 3. Predict verb labels and estimate time to contact (TTC)

[SlowFast](https://github.com/facebookresearch/SlowFast)を用いた行動ラベル抽出と接触時間予測。

## Data

データセットは動画フレーム＋アノテーションの情報が与えられる。1つのitemが1つのイベントに対応する。ただし、アノテーション情報は特定の直前フレームの情報を抽出したものである。

### Annotation

Annotationには以下の情報が含まれる。

#### `fho_sta_<split>_.json`

```json
{
  "description": "...",
  "version": "2.0",
  "split": "train",
  "include_annotations": true,
  "video_metadata": { ... },
  "items": [ ... ]   ← Training sample
}
```

- Contents of `"items"`

```json
{
"uid": "unique_id",
"video_id": "video_uid",
"frame": <frame_number>,
"clip_id": "...",
"clip_uid": "...",
"clip_frame": <frame_index_in_clip>,
"objects": [ ... ]   ← Annotations per objects
}
```

- Contents of `"objects"`
  - The `verb` has 98 classes.
  - The `noun` has 301 classes.

```json
{
  "box": [x1, y1, x2, y2],
  "verb_category_id": <int>,
  "noun_category_id": <int>,
  "ttc": <float>   ← Time-To-Contact
}
```

- Contents of `"video_metadata"`
  It contains full scale resolution. For example, 1920x1080(1080p), 1280x720(720p)

```json
"video_metadata": {
  "<video_uid>": {
    "frame_width": <int>,
    "frame_height": <int>,
    "fps": <float>,
    "year": "...",
    "date_created": "..."
  }
}
```

#### Hand Boxes

手のbboxのアノテーション情報。すべてのフレームに対して左右の手のbbox座標が書かれている。

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

`fho_main.json` is annotation data for forecasting.

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
        }, 
        "5b97f47f-f015-46f3-8879-3fcc2a61a728": {
          "frame_width": 1440, 
          "frame_height": 1080, 
          "fps": 30.0
        }, 
        "3b609b23-f91d-43da-9918-ce928181f53f": {
          "frame_width": 1440, 
          "frame_height": 1080, 
          "fps": 30.0
          }, 
        "9b316b36-7f09-450d-b397-1961723fefb7": {
          "frame_width": 1440, 
          "frame_height": 1080, 
          "fps": 30.0
        }, 
        "7f9f75fd-a660-4635-8890-239c6ad82023": {
          "frame_width": 1440, 
          "frame_height": 1080, 
          "fps": 30.0
        },
```