import itertools
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast, Literal

import numpy as np
import torch
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate  # noqa: PLC2701

from sta_baseline.datasets.build import build_dataset
from sta_baseline.utils.type_alias import Split


def detection_collate(batch: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    """Collate function for detection task.

    Concatenate bboxes, labels and metadata from different samples in the first dimension instead of
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


def _get_loader_settings(cfg: CfgNode, split: Split) -> tuple[str, int, bool, bool]:
    if split == Split.TEST:
        dataset_name = cast("str", cfg.TEST.DATASET)
        batch_size = cast("int", cfg.TEST.BATCH_SIZE)
        return dataset_name, batch_size, False, False

    dataset_name = cast("str", cfg.TRAIN.DATASET)
    batch_size = cast("int", cfg.TRAIN.BATCH_SIZE)
    if cfg.SOLVER.ACCELERATOR != "dp":
        batch_size //= cast("int", cfg.NUM_GPUS)
    return dataset_name, batch_size, split == Split.TRAIN, split == Split.TRAIN


def _get_collate(task: Literal["detection", "short_term_anticipation"]) -> Callable[..., Any] | None:
    if task == "detection":
        return detection_collate
    if task == "short_term_anticipation":
        return sta_collate
    return None


def construct_loader(cfg: CfgNode, split: Split) -> DataLoader:
    """Constructs the data loader for the given dataset.

    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (Split): the split of the data loader. Options include `Split.TRAIN`,
            `Split.VAL`, and `Split.TEST`.
    """
    dataset_name, batch_size, shuffle, drop_last = _get_loader_settings(cfg, split)

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
        collate_fn=_get_collate(cast("Literal['detection', 'short_term_anticipation']", cfg.DATA.TASK)),
    )
    return loader


def sta_collate(batch: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
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
