# sta_baseline

EGO4D Forecasting Short Term Anticipation (STA)

A baseline implementation for developing and benchmarking STA models.

## Table of Contents

- [System Architecture](#system-architecture)
- [Development](#development)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run](#run)
- [Datasets](#datasets)
  - [Download Datasets from EGO4D](#download-datasets-from-ego4d)
  - [Create LMDB](#create-lmdb)

## System Architecture

This baseline is based on [EGO4D Forecasting](https://github.com/EGO4D/forecasting/tree/main).

See [the system architecture](docs/system_architecture.md).

## Development

### Prerequisites

- Python
  - uv 0.9.2
  - See [pyproject.toml](pyproject.toml).
- Tools
  - justfile
  - aws-cli/2.34.32 (required only when downloading datasets from the official EGO4D release)

### Installation

#### Set up the development environment

```bash
git clone git@github.com:shunpcho/sta-baseline.git
uv sync --group dev
source .venv/bin/activate
```

#### Set up VPN

The VPN is required to access the DVC remote. Contact the project maintainer for access.

#### Set up datasets

Annotation files and the LMDB dataset are required for training.

- Download the training LMDB dataset for efficient training. Its total size is approximately 160 GB.

  ```bash
  just dvc_pull_lmdb_data
  ```

- Download the annotation files. They are approximately 2 GB.

  ```bash
  dvc pull ego4d_data/v2/annotations
  ```

- Generate COCO-style annotations.
  - Training annotations

    ```bash
    mkdir -p short_term_anticipation/annotations
    python scripts/create_coco_annotations.py ego4d_data/v2/annotations/fho_sta_train.json short_term_anticipation/annotations/train_coco.json
    ```

  - Validation annotations

    ```bash
    python scripts/create_coco_annotations.py ego4d_data/v2/annotations/fho_sta_val.json short_term_anticipation/annotations/val_coco.json
    ```

- Download the pretrained SlowFast model.

  ```bash
  just dvc_pull_slowfast_model
  ```

### Run

Train the model with the following command:

```bash
mkdir -p short_term_anticipation/models/slowfast_model/
python src/sta_baseline/run_sta.py \
    --cfg config_yaml/SLOWFAST_32x1_8x4_R50_v2.yaml \
    EGO4D_STA.ANNOTATION_DIR ego4d_data/v2/annotations \
    EGO4D_STA.RGB_LMDB_DIR data/clip_lmdb \
    EGO4D_STA.OBJ_DETECTIONS ego4d_data/v2/sta_models/object_detections.json \
    OUTPUT_DIR short_term_anticipation/models/slowfast_model/
```

#### Download pre-trained models and pre-extracted object detections

Pretrained models and pre-extracted object detections are available for use.

## Datasets

### Download Datasets from EGO4D

Skip this procedure if you have access to this project's DVC remote.

- `ego4d_data` contains raw datasets from the official EGO4D release. Download the required data with DVC.
  - The video clips are approximately 160 GB.

See [the dataset download instructions](docs/download_data.md).

### Create LMDB

The baseline recreates the LMDB dataset from the raw data to avoid expensive video processing during training. It contains frame metadata and images associated with timestamps.

- Download dataset from DVC.

```bash
just dvc_pull_raw_data
```

- Run creating LMDB.

```bash
just create_lmdb
```
