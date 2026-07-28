import itertools
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate  # noqa: PLC2701

from sta_baseline.datasets.build import build_dataset
from sta_baseline.utils.type_alias import Split


def detection_collate(batch: tuple | list) -> tuple:
    """Collate function for detection task.

    Concatanate bboxes, labels and metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.

    Args:
        batch (tuple or list): data batch to collate.

    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch, strict=True)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0]:
        data = [d[key] for d in extra_data]
        if key == "gt_labels":
            collated_extra_data[key] = torch.tensor(np.concatenate(data, axis=0)).float()
        elif key in {"boxes", "ori_boxes", "gt_boxes"}:
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate([np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1)
                for i in range(len(data))
                if len(data[i]) > 0
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key in {"metadata", "gt_metadata"}:
            collated_extra_data[key] = list(itertools.chain(*data))
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data


def construct_loader(cfg: CfgNode, split: Split) -> DataLoader:
    """Constructs the data loader for the given dataset.

    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (Split): the split of the data loader. Options include `Split.TRAIN`,
            `Split.VAL`, and `Split.TEST`.
    """
    if split == Split.TRAIN:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = True
    elif split == Split.VAL:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split == Split.TEST:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS) if cfg.SOLVER.ACCELERATOR != "dp" else cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    def get_collate(key: Literal["detection", "short_term_anticipation"]) -> callable | None:
        if key == "detection":
            return detection_collate
        elif key == "short_term_anticipation":
            return sta_collate
        else:
            return None

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training

    sampler = None
    if not cfg.FBLEARNER:
        # Create a sampler for multi-process training
        if hasattr(dataset, "sampler"):
            sampler = dataset.sampler
        elif cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=get_collate(cfg.DATA.TASK),
    )
    return loader


def sta_collate(batch: tuple | list) -> tuple:
    """Collate function for the short term anticipation task.

    Args:
        batch (tuple or list): data batch to collate.

    Returns:
        (tuple): collated detection data batch.
    """
    eids, inputs, pred_boxes, verb_labels, ttc_targets, extra = zip(*batch, strict=True)

    eids = default_collate(eids)
    inputs = default_collate(inputs)

    pred_boxes = [torch.from_numpy(b.astype(float)) for b in pred_boxes]
    verb_labels = [torch.from_numpy(x).long() for x in verb_labels]
    ttc_targets = [torch.from_numpy(x.reshape(-1, 1)).float() for x in ttc_targets]

    extra_data = defaultdict(list)

    for ed in extra:
        for k, v in ed.items():
            extra_data[k].append(v)

    return eids, inputs, pred_boxes, verb_labels, ttc_targets, extra_data
