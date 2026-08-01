#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from collections.abc import Callable

from torch import nn

_LOSSES: dict[str, Callable[..., nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def get_loss_func(loss_name: str) -> Callable[..., nn.Module]:
    """Retrieve the loss given the loss name.

    Args:
        loss_name (str): the name of the loss to use.

    Returns:
        loss_func (Callable[..., nn.Module]): the loss function.

    Raises:
        NotImplementedError: if the loss is not supported.
    """
    if loss_name not in _LOSSES:
        msg = f"Loss {loss_name} is not supported"
        raise NotImplementedError(msg)
    return _LOSSES[loss_name]
