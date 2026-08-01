#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch
from fvcore.common.config import CfgNode

from sta_baseline.optimizers import lr_policy


def construct_optimizer(model: torch.nn.Module, cfg: CfgNode) -> torch.optim.Optimizer:
    """Construct a stochastic gradient descent or ADAM optimizer with momentum.

    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (torch.nn.Module): model to perform stochastic gradient descent
            optimization or ADAM optimization.
        cfg (CfgNode): configs of hyper-parameters of SGD or ADAM, includes base
            learning rate,  momentum, weight_decay, dampening, and etc.

    Returns:
        optimizer (torch.optim.Optimizer): the constructed SGD or ADAM optimizer.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(bn_params), (
        f"parameter size does not match: {len(non_bn_parameters)} + {len(bn_params)} != {len(list(model.parameters()))}"
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"Does not support {cfg.SOLVER.OPTIMIZING_METHOD} optimizer")


def get_epoch_lr(cur_epoch: float, cfg: CfgNode) -> float:
    """Retrieves the lr for the given epoch (as specified by the lr policy).

    Args:
        cfg (CfgNode): configs of hyper-parameters of ADAM, includes base
            learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """Sets the optimizer lr to the specified value.

    Args:
        optimizer (torch.optim.Optimizer): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
