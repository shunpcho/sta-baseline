#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import subprocess
from collections.abc import Iterable, Iterator
from typing import cast

import numpy as np
import psutil
import torch
from fvcore.common.config import CfgNode
from fvcore.nn.flop_count import flop_count
from fvcore.nn.precise_bn import update_bn_stats

from sta_baseline.models.batchnorm_helper import SubBatchNorm3d
from sta_baseline.utils import logging
from sta_baseline.utils.datasets_utils import pack_pathway_output

logger = logging.get_logger(__name__)


def params_count(model: torch.nn.Module) -> int:
    """Compute the number of parameters.

    Args:
        model (model): model to count the number of parameters.
    """
    return int(np.sum([p.numel() for p in model.parameters()]))


def gpu_mem_usage() -> float:
    """Compute the GPU memory usage for the current device (GB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024**3


def cpu_mem_usage() -> tuple[float, float]:
    """Compute the system memory (RAM) usage for the current device (GB).

    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def get_flop_stats(model: torch.nn.Module, cfg: CfgNode, is_train: bool) -> float:
    """Compute the gflops for the current model given the config.

    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.

    Returns:
        float: the total number of gflops of the given model.
    """
    rgb_dimension = 3
    num_frames = cast("int", cfg.DATA.NUM_FRAMES)
    crop_size = cast("int", cfg.DATA.TRAIN_CROP_SIZE if is_train else cfg.DATA.TEST_CROP_SIZE)
    if is_train:
        input_tensors = torch.rand(
            rgb_dimension,
            num_frames,
            crop_size,
            crop_size,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            num_frames,
            crop_size,
            crop_size,
        )
    flop_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(flop_inputs)):
        flop_inputs[i] = flop_inputs[i].unsqueeze(0).cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.
    if cfg.DATA.TASK == "detection":
        bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
        bbox = bbox.cuda()
        inputs = (flop_inputs, bbox)
    else:
        inputs = (flop_inputs,)

    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops


def log_model_info(model: torch.nn.Module, cfg: CfgNode, is_train: bool = True) -> None:
    """Log info, includes number of parameters, gpu usage and gflops.

    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        is_train (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info(f"Model:\n{model}")
    logger.info(f"Params: {params_count(model):,}")
    logger.info(f"Mem: {gpu_mem_usage():,} MB")
    logger.info(f"FLOPs: {get_flop_stats(model, cfg, is_train):,} GFLOPs")
    logger.info("nvidia-smi")
    subprocess.run(["nvidia-smi"], check=False)


def aggregate_split_bn_stats(module: torch.nn.Module) -> int:
    """Recursively find all SubBN modules and aggregate sub-BN stats.

    Args:
        module (torch.nn.Module): module to inspect recursively.

    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_split_bn_stats(child)
    return count


def calculate_and_update_precise_bn(
    loader: Iterable[tuple[torch.Tensor | list[torch.Tensor], ...]],
    model: torch.nn.Module,
    num_iters: int = 200,
) -> None:
    """Update the stats in bn layers by calculate the precise stats.

    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader() -> Iterator[torch.Tensor | list[torch.Tensor]]:
        for inputs, *_ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)
