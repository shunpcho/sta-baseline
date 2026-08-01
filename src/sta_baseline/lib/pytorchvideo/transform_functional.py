# The package, ``pytorchvideo`` has not been maintained since five years ago. The code in this file is copied from the
# original package and modified to fit the needs of this project. The original package can be found at https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/functional.html
# The error below is caused by the original package's dependency on torchvision, which has been removed from this
# project. The code in this file has been modified to remove the dependency on torchvision.

# pytorchvideo/transforms/augmentations.py", line 9, in <module>
#     import torchvision.transforms.functional_tensor as F_t
# ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'

import torch


def uniform_temporal_subsample(x: torch.Tensor, num_samples: int, temporal_dim: int = -3) -> torch.Tensor:
    """Uniformly subsamples num_samples indices from the temporal dimension of the video.

    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0
    assert t > 0

    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)
