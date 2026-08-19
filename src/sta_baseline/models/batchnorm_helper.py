"""BatchNorm (BN) utility functions and custom batch-size BN implementations."""

from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from fvcore.common.config import CfgNode
from torch import nn
from torch.autograd.function import Function

from sta_baseline.utils import distributed as du


def get_norm(cfg: CfgNode) -> type[nn.BatchNorm3d] | partial[Any]:
    """Get the configured normalization layer.

    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.

    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE == "batchnorm":
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == "sub_batchnorm":
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == "sync_batchnorm":
        return partial(NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES)
    else:
        msg = f"Norm type {cfg.BN.NORM_TYPE} is not supported"
        raise NotImplementedError(msg)


class SubBatchNorm3d(nn.Module):
    """The standard BN layer computes stats across all examples in a GPU.

    In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits: int, **args: Any) -> None:  # noqa: ANN401
        """Initialize the split BatchNorm layer.

        Args:
            num_splits (int): number of splits.
            args: BatchNorm keyword arguments.
        """
        super().__init__()
        self.num_splits = num_splits
        num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = num_features * num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(
        self, means: torch.Tensor, stds: torch.Tensor, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the aggregated mean and stds.

        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = stds.view(n, -1).sum(0) / n + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        return mean.detach(), std.detach()

    def aggregate_stats(self) -> None:
        """Synchronize running_mean, and running_var. Call this before eval."""
        if self.split_bn.track_running_stats:
            running_mean = self.bn.running_mean
            running_var = self.bn.running_var
            split_running_mean = self.split_bn.running_mean
            split_running_var = self.split_bn.running_var
            assert running_mean is not None
            assert running_var is not None
            assert split_running_mean is not None
            assert split_running_var is not None
            running_mean.data, running_var.data = self._get_aggregated_mean_std(
                split_running_mean, split_running_var, self.num_splits
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x *= self.weight.view((-1, 1, 1, 1))
            x += self.bias.view((-1, 1, 1, 1))
        return x


class GroupGather(Function):
    """GroupGather performs all gather on each of the local process/ GPU groups."""

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        tensor: torch.Tensor,
        num_sync_devices: int,
        num_groups: int,
    ) -> torch.Tensor:
        """Perform forwarding, gathering the stats across different process/GPU groups."""
        ctx.num_sync_devices = num_sync_devices
        ctx.num_groups = num_groups

        input_list = [torch.zeros_like(tensor) for _ in range(du.get_local_size())]
        dist.all_gather(input_list, tensor, async_op=False, group=du.get_local_process_group())

        inputs = torch.stack(input_list, dim=0)
        if num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // num_sync_devices
            inputs = inputs[group_idx * num_sync_devices : (group_idx + 1) * num_sync_devices]
        inputs = torch.sum(inputs, dim=0)
        return inputs

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Perform backwarding, gathering gradients across different process/GPU groups."""
        grad_output_list = [torch.zeros_like(grad_output) for _ in range(du.get_local_size())]
        dist.all_gather(grad_output_list, grad_output, async_op=False, group=du.get_local_process_group())

        grads = torch.stack(grad_output_list, dim=0)
        if ctx.num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // ctx.num_sync_devices
            grads = grads[group_idx * ctx.num_sync_devices : (group_idx + 1) * ctx.num_sync_devices]
        grads = torch.sum(grads, dim=0)
        return grads, None, None


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_sync_devices: int, **args: Any) -> None:  # noqa: ANN401
        """Naive version of Synchronized 3D BatchNorm.

        Args:
            num_sync_devices (int): number of device to sync.
            args (list): other arguments.
        """
        self.num_sync_devices = num_sync_devices
        super().__init__(**args)

    def _get_num_groups(self) -> int:
        num_groups = 1
        if self.num_sync_devices > 0:
            num_groups = du.get_local_size() // self.num_sync_devices

        return num_groups

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if du.get_local_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        channels = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = torch.as_tensor(GroupGather.apply(vec, self.num_sync_devices, self._get_num_groups()))
        vec *= 1.0 / self.num_sync_devices

        mean, meansqr = torch.split(vec, channels)
        var = meansqr - mean * mean
        running_mean = self.running_mean
        running_var = self.running_var
        momentum = self.momentum
        assert running_mean is not None
        assert running_var is not None
        assert momentum is not None
        running_mean += momentum * (mean.detach() - running_mean)
        running_var += momentum * (var.detach() - running_var)

        invstd = torch.rsqrt(var + self.eps)
        weight = self.weight
        bias_parameter = self.bias
        assert weight is not None
        assert bias_parameter is not None
        scale = weight * invstd
        bias = bias_parameter - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias
