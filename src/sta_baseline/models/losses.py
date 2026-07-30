#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from torch import nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def get_loss_func(loss_name: int) -> callable:
    """Retrieve the loss given the loss name.

    Args:
        loss_name (int): the name of the loss to use.

    Returns:
        loss_func (callable): the loss function.

    Raises:
        NotImplementedError: if the loss is not supported.
    """
    if loss_name not in _LOSSES:
        raise NotImplementedError(f"Loss {loss_name} is not supported")
    return _LOSSES[loss_name]
