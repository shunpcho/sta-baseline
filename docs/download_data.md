# How to Download EGO4D Datasets

Download EGO4D datasets with the [EGO4D CLI](https://github.com/facebookresearch/Ego4d). Video clips, annotations, and pretrained models are available. Use the following commands to download the datasets from the official EGO4D release.

## Table of Contents

- [Prerequisites](#prerequisites)
  - [EGO4D License Agreement](#ego4d-license-agreement)
  - [Set Up AWS](#set-up-aws)
  - [Download Datasets](#download-datasets)
  - [Download the SlowFast Model](#download-the-slowfast-model)
  - [Download STA Pretrained Models](#download-sta-pretrained-models)
- [Next Steps](#next-steps)

## Prerequisites

Install the [EGO4D Dataset Download CLI](https://ego4d-data.org/docs/CLI/). An approved EGO4D license and AWS credentials are required.

### EGO4D License Agreement

Request access through the [EGO4D License Agreement](https://ego4d.dev/request/ego4d). Approval may take approximately two days.

### Set Up AWS

Install the AWS CLI, then configure the access key provided by EGO4D:

```bash
aws configure --profile ego4d
```

### Download Datasets

- STA annotations

  Annotation data in JSON and CSV formats.

- STA clips

  Short video clips extracted for the task.

- Expected directory structure:

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
  --output_directory ego4d_data \
  --datasets annotations,clips \
  --benchmarks fho \
  --aws_profile_name ego4d \
  --version v2
  ```

### Download the SlowFast Model

```bash
mkdir -p data/pretrained_models/
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl -O data/pretrained_models/SLOWFAST_8x8_R50.pkl
```

### Download STA Pretrained Models

Download the STA pretrained models and pre-extracted object detections with the following CLI command:

```bash
ego4d --output_directory="ego4d_data" --datasets sta_models --aws_profile_name ego4d --version v2
```

## Next Steps

Create an LMDB dataset before training to avoid repeatedly reading video files. See [the preprocessing instructions](../README.md#create-lmdb).
