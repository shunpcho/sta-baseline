import itertools
import json
import pathlib
from collections.abc import Iterable

import numpy as np
import torch
from fvcore.common.config import CfgNode
from fvcore.nn.precise_bn import get_bn_modules

import sta_baseline.utils.distributed as du
from sta_baseline.evaluation.sta_evaluate import STAMeanAveragePrecision
from sta_baseline.models import losses
from sta_baseline.tasks.video_task import VideoTask
from sta_baseline.utils import logging, misc
from sta_baseline.utils.type_alias import ShortTermAnticipationBatch

logger = logging.get_logger(__name__)
_IGNORE_VERB_LABEL = -100


def t2a(tensors: Iterable[torch.Tensor]) -> list[np.ndarray]:
    return [t.detach().cpu().numpy().copy() for t in tensors]


class ShortTermAnticipationTask(VideoTask):
    checkpoint_metric = "val/map_box_noun_verb_ttc_err"

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        del self.loss_fun
        self.verb_loss_fun = losses.get_loss_func(cfg.MODEL.VERB_LOSS_FUNC)(reduction="mean")
        self.ttc_loss_fun = losses.get_loss_func(cfg.MODEL.TTC_LOSS_FUNC)(reduction="mean")
        self.lossw: list[int] = cfg.MODEL.STA_LOSS_WEIGHTS
        self._train_epoch_outputs = []
        self._val_epoch_outputs = []
        self._test_epoch_outputs = []

    def training_step(self, batch: ShortTermAnticipationBatch, batch_idx: int) -> dict[str, float | torch.Tensor]:
        inputs = batch.images
        pred_boxes = batch.pred_boxes
        verb_labels = batch.verb_labels
        ttc_targets = batch.ttc_targets

        # model forward pass
        pred_verb, pred_ttc = self.model.forward(inputs, pred_boxes)

        # concatenate verb and ttc targets
        verb_labels = torch.cat(verb_labels, 0)
        verb_labels = verb_labels.to(pred_verb.device)
        ttc_targets = torch.cat(ttc_targets, 0)
        ttc_targets = ttc_targets.to(pred_ttc.device)

        valid_verbs = verb_labels != _IGNORE_VERB_LABEL
        valid_verbs = valid_verbs.to(pred_verb.device)
        verb_labels = verb_labels[valid_verbs]
        pred_verb = pred_verb[valid_verbs]

        verb_loss = self.verb_loss_fun(pred_verb, verb_labels)

        # compute ttc loss only on valid targets
        valid_ttcs = ~torch.isnan(ttc_targets)
        pred_ttc = pred_ttc[valid_ttcs]
        ttc_targets = ttc_targets[valid_ttcs]

        ttc_loss = self.ttc_loss_fun(pred_ttc, ttc_targets)

        loss = self.lossw[0] * verb_loss + self.lossw[1] * ttc_loss

        pred_verb = np.concatenate(du.all_gather_unaligned(pred_verb.detach().cpu().numpy()), axis=0)
        pred_ttc = np.concatenate(du.all_gather_unaligned(pred_ttc.detach().cpu().numpy()), axis=0)

        verb_labels = np.concatenate(du.all_gather_unaligned(verb_labels.detach().cpu().numpy()), axis=0)
        ttc_targets = np.concatenate(du.all_gather_unaligned(ttc_targets.detach().cpu().numpy()), axis=0)

        pred_classes = pred_verb.argmax(-1)
        num_accurate = (pred_classes == verb_labels).sum()
        num_instances = len(verb_labels)
        cls_accuracy = num_accurate / num_instances  # correctly predicted labels

        # mean average error
        ttc_error = np.abs(pred_ttc - ttc_targets).mean()

        step_result = {
            "train/loss": loss.item(),
            "train/verb_loss": verb_loss.item(),
            "train/ttc_loss": ttc_loss.item(),
            "train/ttc_error": ttc_error,
            "train/verb_accuracy": cls_accuracy,
        }

        for key, metric in step_result.items():
            self.log(key, metric)

        step_result["loss"] = loss
        self._train_epoch_outputs.append({k: v for k, v in step_result.items() if "loss" not in k})

        return step_result

    def on_train_epoch_end(self):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE)
        _ = misc.aggregate_split_bn_stats(self.model)

        if not self._train_epoch_outputs:
            return

        keys = list(self._train_epoch_outputs[0].keys())
        for key in keys:
            metric = torch.tensor([x[key] for x in self._train_epoch_outputs]).mean()
            self.log(key, metric)
        self._train_epoch_outputs.clear()

    def validation_step(self, batch, batch_idx):
        uids, inputs, pred_boxes, verb_labels, ttc_targets, extra_data = batch

        # model forward pass
        detections, raw_predictions = self.model.forward(
            inputs,
            pred_boxes,
            extra_data["orig_pred_boxes"],
            extra_data["pred_object_labels"],
            extra_data["pred_object_scores"],
        )

        output = {
            "uids": uids,
            "pred_detections": detections,
            "gt_detections": extra_data["gt_detections"],
            "verb_labels": t2a(verb_labels),
            "ttc_targets": t2a(ttc_targets),
            "verb_scores": [torch.from_numpy(x["verb_scores"]) for x in raw_predictions],
            "ttcs": [torch.from_numpy(x["ttcs"]) for x in raw_predictions],
        }
        self._val_epoch_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self._val_epoch_outputs:
            return

        data = {}
        for k in self._val_epoch_outputs[0]:
            data[k] = list(itertools.chain(*[x[k] for x in self._val_epoch_outputs]))
            data[k] = list(itertools.chain(*du.all_gather_unaligned(data[k])))

        _, unique_idx = np.unique(data["uids"], return_index=True)

        # remove duplicates
        for k in data:
            data[k] = [data[k][i] for i in unique_idx]

        metric = STAMeanAveragePrecision()
        for prediction, ground_truth in zip(data["pred_detections"], data["gt_detections"], strict=True):
            metric.add(prediction, ground_truth)

        vals = metric.evaluate()
        names = metric.get_short_names()

        for name, val in zip(names, vals, strict=True):
            self.log(f"val/{name}", val)

        self.log("val/map_box_noun_verb_ttc_err", 100 - vals[-1])

        pred_verb = np.concatenate(data["verb_scores"], axis=0)
        pred_ttc = np.concatenate(data["ttcs"], axis=0)

        verb_labels = np.concatenate(data["verb_labels"], axis=0)
        ttc_targets = np.concatenate(data["ttc_targets"], axis=0)

        valid_verbs = verb_labels != _IGNORE_VERB_LABEL
        verb_labels = verb_labels[valid_verbs]
        pred_verb = pred_verb[valid_verbs]

        pred_classes = pred_verb.argmax(-1)
        num_accurate = (pred_classes == verb_labels).sum()
        num_instances = len(verb_labels)
        cls_accuracy = num_accurate / num_instances  # correctly predicted labels

        valid_ttcs = ~np.isnan(ttc_targets)
        pred_ttc = pred_ttc[valid_ttcs]
        ttc_targets = ttc_targets[valid_ttcs]

        # mean average error
        ttc_error = np.abs(pred_ttc - ttc_targets).mean()

        self.log("val/ttc_error", ttc_error)
        self.log("val/verb_accuracy", cls_accuracy)
        self._val_epoch_outputs.clear()

    def test_step(self, batch, batch_idx):
        uids, inputs, pred_boxes, _, _, extra_data = batch

        # model forward pass
        detections, _ = self.model.forward(
            inputs,
            pred_boxes,
            extra_data["orig_pred_boxes"],
            extra_data["pred_object_labels"],
            extra_data["pred_object_scores"],
        )

        output = {"uids": uids, "pred_detections": detections}
        self._test_epoch_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        if not self._test_epoch_outputs:
            return

        data = {}
        for k in self._test_epoch_outputs[0]:
            data[k] = list(itertools.chain(*[x[k] for x in self._test_epoch_outputs]))
            data[k] = list(itertools.chain(*du.all_gather_unaligned(data[k])))

        _, unique_idx = np.unique(data["uids"], return_index=True)

        # remove duplicates
        for k in data:
            data[k] = [data[k][i] for i in unique_idx]

        if self.global_rank == 0:
            res = {
                "version": "1.0",
                "challenge": "ego4d_short_term_object_interaction_anticipation",
                "results": {
                    uid: [
                        {
                            "box": [float(h) for h in z[0]],
                            "noun_category_id": int(z[1]),
                            "verb_category_id": int(z[2]),
                            "time_to_contact": float(z[3]),
                            "score": float(z[4]),
                        }
                        for z in zip(x["boxes"], x["nouns"], x["verbs"], x["ttcs"], x["scores"], strict=True)
                    ]
                    for uid, x in zip(data["uids"], data["pred_detections"], strict=True)
                },
            }
            with pathlib.Path(self.cfg.RESULTS_JSON).open("w") as f:
                json.dump(res, f)

        self._test_epoch_outputs.clear()
