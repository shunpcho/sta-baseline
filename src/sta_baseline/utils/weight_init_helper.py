#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Utility functions for weight initialization."""

from typing import cast

from fvcore.nn.weight_init import c2_msra_fill
from torch import nn, Tensor


def init_weights(model: nn.Module, fc_init_std: float = 0.01, zero_init_final_bn: bool = True) -> None:
    """Performs ResNet style weight initialization.

    Args:
        model (nn.Module): model whose weights to initialize.
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            if hasattr(m, "transform_final_bn") and m.transform_final_bn and zero_init_final_bn:
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            batchnorm_weight_parameter = cast("Tensor | None", m.weight)
            batchnorm_bias_parameter = cast("Tensor | None", m.bias)
            if batchnorm_weight_parameter is not None:
                batchnorm_weight_parameter.data.fill_(batchnorm_weight)
            if batchnorm_bias_parameter is not None:
                batchnorm_bias_parameter.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()
