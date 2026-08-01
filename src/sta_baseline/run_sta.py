import pickle
import pprint
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast, Protocol

import numpy as np
import torch
from fvcore.common.config import CfgNode
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from sta_baseline.tasks.short_term_anticipation import ShortTermAnticipationTask
from sta_baseline.utils import logging
from sta_baseline.utils.c2_model_loading import get_name_convert_func
from sta_baseline.utils.parser import load_config, parse_args

logger = logging.get_logger(__name__)


class _BackboneModel(Protocol):
    backbone: torch.nn.Module


def _get_task(cfg: CfgNode) -> ShortTermAnticipationTask:
    if cfg.DATA.TASK == "short_term_anticipation":
        return ShortTermAnticipationTask(cfg)
    msg = f"Unsupported task: {cfg.DATA.TASK}"
    raise ValueError(msg)


def _remove_module_prefix(key: str) -> str:
    return ".".join(key.split(".")[1:])


def _load_caffe2_checkpoint(task: ShortTermAnticipationTask, checkpoint_path: str, load_model_head: bool) -> None:
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        data = cast("Mapping[str, Mapping[str, Any]]", pickle.load(checkpoint_file, encoding="latin1"))

    convert_name = cast("Callable[[str], str]", get_name_convert_func())
    state_dict = {
        convert_name(key): torch.from_numpy(np.asarray(value))
        for key, value in data["blobs"].items()
        if not any(excluded in key for excluded in ("momentum", "lr", "model_iter"))
    }
    if not load_model_head:
        state_dict = {key: value for key, value in state_dict.items() if "head" not in key}

    logger.info("%s", task.model.load_state_dict(state_dict, strict=False))
    logger.info("Checkpoint %s loaded", checkpoint_path)


def _load_backbone_checkpoint(task: ShortTermAnticipationTask, checkpoint_path: str) -> None:
    checkpoint = cast("Mapping[str, Mapping[str, torch.Tensor]]", torch.load(checkpoint_path, map_location="cpu"))
    state_dict = {
        _remove_module_prefix(key): value for key, value in checkpoint["state_dict"].items() if "head" not in key
    }
    model = cast("_BackboneModel", task.model)
    load_result = model.backbone.load_state_dict(state_dict, strict=False)
    missing_keys, unexpected_keys = cast("tuple[list[str], list[str]]", load_result)
    if unexpected_keys or any("head" not in key for key in missing_keys):
        msg = f"Unexpected backbone checkpoint keys: {unexpected_keys}; missing keys: {missing_keys}"
        raise RuntimeError(msg)


def _load_lightning_checkpoint(task: ShortTermAnticipationTask, checkpoint_path: str, load_model_head: bool) -> None:
    pretrained = ShortTermAnticipationTask.load_from_checkpoint(checkpoint_path)
    states_by_child = {child_name: child.state_dict() for child_name, child in pretrained.model.named_children()}
    for child_name, child in task.model.named_children():
        if not load_model_head and "head" in child_name:
            continue
        child.load_state_dict(states_by_child[child_name])


def _load_checkpoint(task: ShortTermAnticipationTask, cfg: CfgNode) -> None:
    checkpoint_path = cast("str", cfg.CHECKPOINT_FILE_PATH)
    if not checkpoint_path:
        return

    if cfg.CHECKPOINT_VERSION == "caffe2":
        _load_caffe2_checkpoint(task, checkpoint_path, cast("bool", cfg.CHECKPOINT_LOAD_MODEL_HEAD))
    elif checkpoint_module_path := cast("str", cfg.DATA.CHECKPOINT_MODULE_FILE_PATH):
        _load_backbone_checkpoint(task, checkpoint_module_path)
    else:
        _load_lightning_checkpoint(task, checkpoint_path, cast("bool", cfg.CHECKPOINT_LOAD_MODEL_HEAD))


def _create_trainer(task: ShortTermAnticipationTask, cfg: CfgNode) -> Trainer:
    checkpoint_callback = ModelCheckpoint(monitor=task.checkpoint_metric, mode="min", save_last=True, save_top_k=1)
    devices = cast("int", cfg.NUM_GPUS)
    strategy = DDPStrategy(find_unused_parameters=False) if devices > 1 else "auto"
    return Trainer(
        accelerator=cast("str", cfg.SOLVER.ACCELERATOR),
        devices=devices,
        num_nodes=cast("int", cfg.NUM_SHARDS),
        strategy=strategy,
        max_epochs=cast("int", cfg.SOLVER.MAX_EPOCH),
        num_sanity_val_steps=3,
        benchmark=True,
        use_distributed_sampler=False,
        fast_dev_run=cast("bool", cfg.FAST_DEV_RUN),
        default_root_dir=cast("str", cfg.OUTPUT_DIR),
        logger=cast("bool", cfg.ENABLE_LOGGING),
        callbacks=[LearningRateMonitor(), checkpoint_callback],
    )


def main(cfg: CfgNode) -> list[Mapping[str, float]] | None:
    seed_everything(cast("int", cfg.RNG_SEED))

    logging.setup_logging(cast("str", cfg.OUTPUT_DIR))
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    task = _get_task(cfg)
    _load_checkpoint(task, cfg)
    trainer = _create_trainer(task, cfg)

    if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)

        # Calling test without the lightning module arg automatically selects the best
        # model during training.
        return trainer.test()

    elif cfg.TRAIN.ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        return trainer.test(task)

    return None


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    main(cfg)
