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
