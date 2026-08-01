from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from torch.utils.data import Dataset

from sta_baseline.utils.type_alias import Split

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name: str, cfg: CfgNode, split: Split) -> Dataset:
    """Build a dataset, defined by `dataset_name`.

    Args:
        dataset_name: the name of the dataset to be constructed.
        cfg: configs. Details can be found in
            ego4d/config/defaults.py
        split: the split of the data loader. Options include `Split.TRAIN`,
            `Split.VAL`, and `Split.TEST`.

    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASET_REGISTRY.get(dataset_name)(cfg, split)
