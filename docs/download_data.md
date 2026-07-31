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
      │   └── fho_sta_test_unannotated.json
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
  --aws_profile_name ego4d \
  --version v2
  ```

- Preprocessing

毎回動画を読み込み学習するのは遅延が大きい。事前にLMDB化し効率よく学習できるようにする。

## Pre-trained models

The pre-trained models and pre-extracted object detections can be downloaded using the CLI with the following command:

```bash
ego4d --output_directory="ego4d_data" --datasets sta_models --aws_profile_name ego4d --version v2
```

### Generating COCO-style annotations

train

```bash
mkdir short_term_anticipation/annotations
python scripts/create_coco_annotations.py ego4d_data/v2/annotations/fho_sta_train.json short_term_anticipation/annotations/train_coco.json
```

val

```bash
python scripts/create_coco_annotations.py ego4d_data/v2/annotations/fho_sta_val.json short_term_anticipation/annotations/val_coco.json
```

### SlowFast model

```bash
mkdir data/pretrained_models/
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl -O data/pretrained_models/SLOWFAST_8x8_R50.pkl
```

## Run

```bash
mkdir -p short_term_anticipation/models/slowfast_model/
python src/sta_baseline/run_sta.py \
    --cfg config_yaml/SLOWFAST_32x1_8x4_R50_v2.yaml \
    EGO4D_STA.ANNOTATION_DIR ego4d_data/v2/annotations \
    EGO4D_STA.RGB_LMDB_DIR data/clip_lmdb \
    EGO4D_STA.OBJ_DETECTIONS ego4d_data/v2/sta_models/object_detections.json
    OUTPUT_DIR short_term_anticipation/models/slowfast_model/
```
