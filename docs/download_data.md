# How to download EGO4D datasets

## Prerequisites

### Ego4D License Agreement

- [ライセンス申請](https://ego4d.dev/request/ego4d)をする(承認まで2日程度)

### Setup AWS

- Install AWS CLI

- Config access key

  ライセンス認証後に送られてくるキーを入力する

  ```bash
  aws configure --profile ego4d
  ```

### Download Datasets

- STA annotations

  JSONとCSVファイルのアノテーションデータ

- STA clips

  STA 用に切り出された短い動画クリップ

- Directory

  ```
  ego4d_data/
  └── v2/
      ├── annotations/
      │   ├── fho_sta_train.json
      │   ├── fho_sta_val.json
      │   └── fho_sta_test.json
      └── clips/
          └── fho_sta/
              ├── clip_000001.mp4
              ├── clip_000002.mp4
              └── ...
  ```

- Download annotations + clips for STA

  ```bash
  ego4d \
  --output_directory ~/ego4d_data \
  --datasets annotations,clips \
  --benchmarks fho \
  --aws_profile_name ego4d
  ```

- Preprocessing

毎回動画を読み込み学習するのは遅延が大きい。事前にLMDB化し効率よく学習できるようにする。

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
      }
    }
  }
}
```
