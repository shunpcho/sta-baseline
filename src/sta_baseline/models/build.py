"""Model construction functions."""

import torch
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg: CfgNode) -> torch.nn.Module:
    """Builds the video model.

    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in ego4d/config/defaults.py.
    """
    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    return model
