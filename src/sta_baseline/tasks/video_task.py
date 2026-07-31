from typing import Any

from fvcore.common.config import CfgNode
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader

from sta_baseline.datasets import loader
from sta_baseline.models import losses
from sta_baseline.models.build import build_model
from sta_baseline.optimizers import lr_scheduler
from sta_baseline.utils import distributed as du
from sta_baseline.utils import logging

logger = logging.get_logger(__name__)


class VideoTask(LightningModule):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()

        # Backwards compatibility.
        if isinstance(cfg.MODEL.NUM_CLASSES, int):
            cfg.MODEL.NUM_CLASSES = [cfg.MODEL.NUM_CLASSES]

        if not hasattr(cfg.TEST, "NO_ACT"):
            logger.info("Default NO_ACT")
            cfg.TEST.NO_ACT = False

        if not hasattr(cfg.MODEL, "TRANSFORMER_FROM_PRETRAIN"):
            cfg.MODEL.TRANSFORMER_FROM_PRETRAIN = False

        if not hasattr(cfg.MODEL, "STRIDE_TYPE"):
            cfg.EPIC_KITCHEN.STRIDE_TYPE = "constant"

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = build_model(cfg)
        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="mean")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR == "dp":
            training_step_outputs["loss"] = training_step_outputs["loss"].mean()
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        # if not self.cfg.FBLEARNER:
        if self.cfg.SOLVER.ACCELERATOR != "dp":
            du.init_distributed_groups(self.cfg)

        self.train_loader: DataLoader[Any] = loader.construct_loader(self.cfg, "train")
        self.val_loader: DataLoader[Any] = loader.construct_loader(self.cfg, "val")
        self.test_loader: DataLoader[Any] = loader.construct_loader(self.cfg, "test")

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY)

    def train_dataloader(self) -> DataLoader[Any]:
        return self.train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        return self.val_loader

    def test_dataloader(self) -> DataLoader[Any]:
        return self.test_loader

    def on_after_backward(self) -> None:
        if self.cfg.LOG_GRADIENT_PERIOD >= 0 and self.trainer.global_step % self.cfg.LOG_GRADIENT_PERIOD == 0:
            for name, weight in self.model.named_parameters():
                if weight is not None:
                    self.logger.experiment.add_histogram(name, weight, self.trainer.global_step)
                    if weight.grad is not None:
                        self.logger.experiment.add_histogram(f"{name}.grad", weight.grad, self.trainer.global_step)
