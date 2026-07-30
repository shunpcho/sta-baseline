#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
from torch import nn

from sta_baseline.lib.detectron2.roi_align import ROIAlign


class ResNetRoIHead(nn.Module):
    """ResNe(X)t RoI head."""

    def __init__(
        self,
        dim_in: list[int],
        num_classes: int,
        pool_size: list[int],
        resolution: list[int],
        scale_factor: list[float],
        dropout_rate: float = 0.0,
        act_func: str = "softmax",
        aligned: bool = True,
    ) -> None:
        """The `__init__` method of any subclass should also contain these arguments.

        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.

        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super().__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d([pool_size[pathway][0], 1, 1], stride=1)
            self.add_module(f"s{pathway}_tpool", temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module(f"s{pathway}_roi", roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module(f"s{pathway}_spool", spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} is not supported as an activationfunction.")

    def forward(self, inputs: list[torch.Tensor], bboxes: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == self.num_pathways, f"Input tensor does not contain {self.num_pathways} pathway"
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, f"s{pathway}_tpool")
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, f"s{pathway}_roi")
            out = roi_align(out, bboxes)

            s_pool = getattr(self, f"s{pathway}_spool")
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """ResNe(X)t 3D head.

    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in: list[int],
        num_classes: int,
        pool_size: list[int | None],
        dropout_rate: float = 0.0,
        act_func: str | None = "softmax",
    ) -> None:
        """The `__init__` method of any subclass should also contain these arguments.

        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super().__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module(f"pathway{pathway}_avgpool", avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func is None:
            self.act = None
        else:
            raise NotImplementedError(f"{act_func} is not supported as an activationfunction.")

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == self.num_pathways, f"Input tensor does not contain {self.num_pathways} pathway"
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, f"pathway{pathway}_avgpool")
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training and self.act is not None:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


# For LTA models. One head per future action prediction
class MultiTaskHead(nn.Module):
    def __init__(
        self,
        dim_in: list[int],
        num_classes: list[int],
        pool_size: list[int | None],
        dropout_rate: float = 0.0,
        act_func: str = "softmax",
        test_noact: bool = False,
    ) -> None:
        super().__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            avg_pool = (
                nn.AvgPool3d(pool_size[pathway], stride=1)
                if pool_size[pathway] is not None
                else nn.AdaptiveAvgPool3d((1, 1, 1))
            )
            self.add_module(f"pathway{pathway}_avgpool", avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = [nn.Linear(sum(dim_in), n, bias=True) for n in num_classes]
        self.projections = nn.ModuleList(projs)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} is not supported as an activationfunction.")

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(inputs) == self.num_pathways, f"Input tensor does not contain {self.num_pathways} pathway"
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, f"pathway{pathway}_avgpool")
            pool_out.append(m(inputs[pathway]))

        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        feat = x
        if hasattr(self, "dropout"):
            feat = self.dropout(feat)

        x = []
        for projection in self.projections:
            # print(feat.shape, projection)
            x.append(projection(feat))

        # Performs fully convlutional inference.
        if not self.training:
            if not self.test_noact:
                x = [self.act(x_i) for x_i in x]
            x = [x_i.mean([1, 2, 3]) for x_i in x]

        x = [x_i.view(x_i.shape[0], -1) for x_i in x]
        return x


class MultiTaskMViTHead(nn.Module):
    def __init__(
        self,
        dim_in: list[int],
        num_classes: list[int],
        dropout_rate: float = 0.0,
        act_func: str = "softmax",
    ) -> None:
        super().__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = [nn.Linear(sum(dim_in), n, bias=True) for n in num_classes]
        self.projections = nn.ModuleList(projs)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} is not supported as an activationfunction.")

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        # Perform dropout.

        feat = inputs
        if hasattr(self, "dropout"):
            feat = self.dropout(feat)

        x = [self.act(projection(feat)) for projection in self.projections]

        return x
