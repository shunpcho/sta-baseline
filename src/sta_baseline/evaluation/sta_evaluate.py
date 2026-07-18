from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt


class BoundingBox(NamedTuple):
    """A bounding box represented by its coordinates."""

    x_1: int
    y_1: int
    x_2: int
    y_2: int


class PredictionsFormat(TypedDict):
    boxes: npt.NDArray[np.uint16]
    scores: npt.NDArray[np.float32]
    nouns: npt.NDArray[np.uint16]
    verbs: npt.NDArray[np.uint16]
    ttcs: npt.NDArray[np.float32]


class Prediction(TypedDict):
    """A single prediction represented by its properties.

    Represents a single prediction with its bounding boxes, scores, and associated labels.
    PredictionsFormat = {
        "boxes": Prediction["box"], Prediction["box"], ...,
        "scores": Prediction["score"], Prediction["score"], ...,
        "nouns": Prediction["noun"], Prediction["noun"], ...,
        "verbs": Prediction["verb"], Prediction["verb"], ...,
        "ttcs": Prediction["ttc"], Prediction["ttc"], ...,
        }
    """

    box: npt.NDArray[np.uint16]
    score: npt.NDArray[np.float32]
    noun: npt.NDArray[np.uint16]
    verb: npt.NDArray[np.uint16]
    ttc: npt.NDArray[np.float32]


class LabelsFormat(TypedDict):
    boxes: npt.NDArray[np.uint16]
    nouns: npt.NDArray[np.uint16]
    verbs: npt.NDArray[np.uint16]
    ttcs: npt.NDArray[np.float32]


def compute_iou(
    preds: list[list[int]] | npt.NDArray[np.int32] | npt.NDArray[np.int64],
    gts: list[list[int]] | npt.NDArray[np.int32] | npt.NDArray[np.int64],
    eps: float = 1e-11,
) -> npt.NDArray[np.float64]:
    """Compute a matrix of intersection over union (IoU) values for two lists of bounding boxes using broadcasting.

    Formula for IoU:
        IoU = Area of Intersection / Area of Union
            = (A ∩ B) / (A u B)

    Args:
        preds: Matrix of predicted bounding boxes with shape (NP, 4).
        gts: Matrix of ground truth bounding boxes with shape (NG, 4).
        eps: A small value to avoid division by zero.

    Returns:
        A matrix of IoU values with shape (NP, NG).
    """
    preds = np.array(preds, dtype=np.int64)
    gts = np.array(gts, dtype=np.int64)

    preds = np.expand_dims(preds, axis=1)  # Shape: (NP, 1, 4)
    gts = np.expand_dims(gts, axis=0)  # Shape: (1, NG, 4)

    def compute_area(boxes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Compute the areas.

        Args:
            boxes: Matrix of bounding boxes.

        Returns:
            The areas.
        """
        widths = boxes[..., 2] - boxes[..., 0] + 1
        heights = boxes[..., 3] - boxes[..., 1] + 1
        widths[widths < 0] = 0
        heights[heights < 0] = 0

        return widths * heights

    ixmin = np.maximum(preds[..., 0], gts[..., 0])
    iymin = np.maximum(preds[..., 1], gts[..., 1])
    ixmax = np.minimum(preds[..., 2], gts[..., 2])
    iymax = np.minimum(preds[..., 3], gts[..., 3])

    areas_preds = compute_area(preds)
    areas_gts = compute_area(gts)
    areas_intersection = compute_area(np.stack([ixmin, iymin, ixmax, iymax], axis=-1))

    iou = areas_intersection / (areas_preds + areas_gts - areas_intersection + eps)

    return iou


class AbstractMeanAveragePrecision(ABC):
    """Abstract class for implementing mean average precision (mAP) measurement."""

    def __init__(
        self,
        num_aps: int,
        percentage: bool = True,
        count_all_classes: bool = True,
        top_k: int | None = None,
    ) -> None:
        """Contruct the mAP metric.

        Args:
            num_aps: Number of average precision metrics to compute. E.g., we can compute different APs for
                    different IoU thresholds.
            percentage: Whether to count all classes when computing the mAP. If false, classes which do not have
                        any ground truth label but do have associated predictions are counted (they will have an
                        AP equal to 0), otherwise, only classes for which there is at least one ground truth label
                        will be counted. It is useful to set this to True for imbalanced datasets for which not all
                        classes are in the ground truth labels.
            count_all_classes: Whether to count all classes when computing the mAP.
            top_k: The K to be considered in the top-k criterion. If None, a standard mAP will be computed.
        """
        self.true_positives: list[npt.NDArray[np.float64]] = []
        self.confidence_scores: list[npt.NDArray[np.float64]] = []
        self.predicted_classes: list[npt.NDArray[np.float64]] = []
        self.gt_classes: list[npt.NDArray[np.float64]] = []

        self.num_aps = num_aps
        self.percentage = percentage
        self.count_all_classes = count_all_classes
        self.top_k = top_k
        self.names: list[str] = []
        self.short_names: list[str] = []

    def get_names(self) -> list[str]:
        return self.names

    def get_short_names(self) -> list[str]:
        return self.short_names

    def add(self, preds: PredictionsFormat, labels: LabelsFormat) -> npt.NDArray[np.float64]:
        """Add predictions and labels of a single image and matches predictions to ground truth boxes.

        Args:
            preds: Dictionary of predictions following the PredictionsFormat. While "boxes" and "scores" are mandatory,
                    other properties can be added (they can be used to compute matchings). It can also be a
                    list of dictionaries, if predictions of more than one image are being added.
            labels: Dictionary of ground truth labels following the LabelsFormat. It can be a list of dictionaries.

        Returns:
            A list of pairs of predicted/matched ground truth boxes.
        """
        matched: list[npt.NDArray[np.float64]] = []

        if len(preds) > 0:
            predicted_boxes = np.array(preds["boxes"], dtype=np.int32)
            predicted_scores = np.array(preds["scores"], dtype=np.float64)
            predicted_classes = self._map_classes(preds)

            # Keep track of correctly matched boxes for the different AP matrics.
            true_positives = np.zeros((len(predicted_boxes), self.num_aps), dtype=np.float64)

            if len(labels) > 0:
                gt_boxes = np.array(labels["boxes"], dtype=np.int32)

                ious = compute_iou(predicted_boxes, gt_boxes)

                # Keep track of GT boxes which have already been matched to a predicted box.
                gt_matched = np.zeros((len(gt_boxes), self.num_aps), dtype=np.float64)

                # From highest to lowest score
                for i in predicted_scores.argsort()[::-1]:
                    # Get overlaps related to this predictions
                    overlaps = ious[i].reshape(-1, 1)

                    pred_dict = Prediction(
                        box=np.asarray(preds["boxes"])[i],
                        score=np.asarray(preds["scores"])[i],
                        noun=np.asarray(preds["nouns"])[i],
                        verb=np.asarray(preds["verbs"])[i],
                        ttc=np.asarray(preds["ttcs"])[i],
                    )
                    matchings = self._match(pred_dict, labels, overlaps)

                    # Replicate overlaps to match shape of matchings (different AP metrics).
                    overlaps = np.tile(overlaps, [1, matchings.shape[1]])

                    # Don't allow to match a matched GT boxes.
                    matchings[gt_matched == 1] = 0

                    # Remove overlaps corresponding to boxes which are not matched.
                    overlaps[matchings == 0] = -1

                    # Get indexes of maximum wrt GT.
                    jj = overlaps.argmax(axis=0)

                    # Get values of matching obtained at the maximum.
                    # These indicate if the matchings are correct or not.
                    i_matchings = matchings[jj, range(len(jj))]

                    jj_matched = jj.copy()
                    jj_matched[~i_matchings] = -1

                    # Set true positive to 1 if we obtained a matching.
                    true_positives[i, i_matchings] = 1

                    # Set the GT asa matched if we obtained a matching.
                    gt_matched[jj, range(len(jj))] += i_matchings

                    matched.append(jj_matched.astype(np.float64))

                # Remove the K highest score false positives if top_k is set.
                if self.top_k is not None and self.top_k > 1:
                    k_full = (self.top_k - 1) * len(labels["boxes"])
                    # Sort the true positives by score
                    order = predicted_scores.argsort()[::-1]
                    sorted_tp = true_positives[order, :].astype(float)

                    sorted_fp = 1 - sorted_tp

                    sorted_tp[(sorted_fp.cumsum(axis=0) <= k_full) & (sorted_fp == 1)] = np.nan

                    true_positives = sorted_tp
                    predicted_scores = predicted_scores[order]
                    predicted_classes = predicted_classes[order]

                self.gt_classes.append(self._map_classes(labels))

            # Append the list of true positives and the confidence scores.
            self.true_positives.append(true_positives)
            self.confidence_scores.append(predicted_scores)
            self.predicted_classes.append(predicted_classes)

        elif len(labels) > 0:
            self.gt_classes.append(self._map_classes(labels))
        if len(matched) > 0:
            return np.vstack(matched)
        else:
            return np.zeros((0, self.num_aps), dtype=np.float64)

    def _map_classes(self, preds_labels: PredictionsFormat | LabelsFormat) -> npt.NDArray[np.float64]:
        """Maps the class labels to a numpy array. This is useful for computing the mAP metric.

        Args:
            preds_labels: The predictions or labels containing class labels.

        Returns:
            The mapped class labels as a numpy array.
        """
        return np.vstack([preds_labels["nouns"]] * self.num_aps).astype(np.float64).T

    def _compute_precision_recall(
        self,
        true_positives: npt.NDArray[np.float64],
        confidence_scores: npt.NDArray[np.float64],
        num_gt: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute precision and recall values.

        Args:
            true_positives: A numpy array of true positives.
            confidence_scores: A numpy array of confidence scores.
            num_gt: The number of ground truth labels for the current class.

        Returns:
            Precision and recall values as numpy arrays.
        """
        # Sort true positives by confidence scores.
        sorted_tp = true_positives[confidence_scores.argsort()[::-1]].astype(np.float64)

        tp = sorted_tp.cumsum()
        fp = (1.0 - sorted_tp).cumsum()

        precision = self._safe_divide(tp, tp + fp)
        recall = self._safe_divide(tp, num_gt)
        return precision, recall

    @staticmethod
    def _safe_divide(
        numerator: npt.NDArray[np.float64], denominator: int | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Safely divide to avoid division by zero.

        Args:
            numerator: The numerator array.
            denominator: The denominator array.

        Returns:
            The result of the division, with zeros where the denominator is zero.
        """
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator != 0)

    def _compute_average_precision(self, precision: npt.NDArray[np.float64], recall: npt.NDArray[np.float64]) -> float:
        """Compute average precision (AP) from precision and recall values.

        Python implementation of Matlab's VOC AP computation.

        Args:
            precision: A numpy array of precision values.
            recall: A numpy array of recall values.

        Returns:
            The average precision value as a float.
        """
        # Append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 2, 0, -1):
            mpre[i] = np.maximum(mpre[i], mpre[i + 1])

        # Look for points where the recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1

        # Sum (\Delta recall) * prec
        ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap

    @staticmethod
    def _compute_max_recall(recall: npt.NDArray[np.float64]) -> float:
        """Compute the maximum recall value.

        Args:
            recall: A numpy array of recall values.

        Returns:
            The maximum recall value.
        """
        max_recall = np.max(recall)
        return float(max_recall)

    def evaluate(self, measure: str = "AP") -> tuple[float, ...]:
        """Evaluate the average precision (AP) or maximum recall (MR) based on the added predictions and labels.

        Args:
            measure: The metric to evaluate. Can be "AP" for average precision or "MR" for maximum recall.

        Returns:
            Returns a tuple containing the evaluated metric values.

        Raises:
            ValueError: If an unknown measure is provided. Supported measures are "AP" and "MR".
        """
        metrics: list[float] = []
        # Compute the different values of the metric for each class.

        gt_classes = np.concatenate(self.gt_classes, axis=0)
        predicted_classes = np.concatenate(self.predicted_classes, axis=0)
        true_positives = np.concatenate(self.true_positives, axis=0)
        confidence_scores = np.concatenate(self.confidence_scores, axis=0)

        for i in range(self.num_aps):
            # The different per-class AP values.
            measures: list[npt.NDArray[np.float32] | float] = []

            gt_classes_i = gt_classes[:, i]
            predicted_classes_i = predicted_classes[:, i]
            true_positives_i = true_positives[:, i]

            if self.count_all_classes:
                classes = np.unique(np.concatenate([gt_classes_i, predicted_classes_i]))
            else:
                classes = np.unique(gt_classes_i)

            # Iterate over each class to compute the metric.
            for cla in classes:
                # Get the true positives and number of ground truth labels.
                true_positive = true_positives_i[predicted_classes_i == cla]
                confidence_score = confidence_scores[predicted_classes_i == cla]
                num_gt = int(np.sum(gt_classes_i == cla))  # type: ignore[reportUnknownArgumentType]

                # Check if the list of true positives is non-empty.
                if len(true_positive) > 0:
                    # Remove the true positives and related confidence scores.
                    valid = ~np.isnan(true_positive)
                    true_positive = true_positive[valid]
                    confidence_score = confidence_score[valid]

                this_measure: float = 0.0
                # If both true positives and number of ground truth are non empty, compute the metric.
                if len(true_positive) > 0 and num_gt > 0:
                    precision, recall = self._compute_precision_recall(true_positive, confidence_score, num_gt)
                    if measure == "AP":
                        this_measure = self._compute_average_precision(precision, recall)
                    elif measure == "MR":
                        this_measure = self._compute_max_recall(recall)
                    else:
                        raise ValueError(f"Unknown measure: {measure}. Supported measures are 'AP' and 'MR'.")

                    # Turn into percentage if required.
                    if self.percentage:
                        this_measure *= 100

                    # Append the computed metric for the list.
                    measures.append(this_measure)

                elif not (len(true_positive) == 0 and num_gt == 0):
                    # If there are no true positives but there are ground truth labels, the metric is 0.
                    measures.append(0.0)

            # Append the mAP value.
            metrics.append(float(np.mean(np.array(measures))))

        # Return the single value or the list of values for each AP metric.
        return tuple(metrics)

    @abstractmethod
    def _match(
        self, pred: Prediction, gt_predictions: LabelsFormat, ious: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Abstract method to return matrix of number of match predictions to ground truth boxes."""


class ObjectOnlyMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Class for computing mean average precision (mAP) based on object detection only.

    This will compute the following metrics:
        - Box + Noun
        - Box
    """

    def __init__(self, iou_threshold: float = 0.5, top_k: int = 3, count_all_classes: bool = False) -> None:
        """Construct the object-only mAP metric.

        Args:
            iou_threshold: The IoU threshold to consider a prediction as a true positive.
            top_k: The K to be considered in the top-k criterion. If None, a standard mAP will be computed.
            count_all_classes: Whether to count all classes when computing the mAP.
        """
        super().__init__(num_aps=2, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.names = ["Box + Noun mAP", "Box AP"]
        self.short_names = ["map_box_noun", "ap_box"]

    def _map_classes(self, preds_labels: PredictionsFormat | LabelsFormat) -> npt.NDArray[np.float64]:
        """Associates the prediction to a class.

        Args:
            preds_labels: The predictions or labels containing class labels.

        Returns:
            An array of class indices corresponding to the predictions.
        """
        nouns = np.array(preds_labels["nouns"], dtype=np.float64)
        boxes = np.ones(len(nouns), dtype=np.float64)

        return np.vstack(
            [
                nouns,  # box + noun, average over nouns
                boxes,  # box, just compute a single AP
            ]
        ).T

    def _match(
        self, pred: Prediction, gt_predictions: LabelsFormat, ious: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Return matches of a given prediction to set of ground truth predictions.

        Args:
            pred: The prediction to match.
            gt_predictions: The ground truth label to match against.
            ious: A matrix of IoU values between the predictions and ground truth boxes.

        Returns:
            A matrix of shape (num_pred, num_aps), specifying possible matchings depending on the prediction and metric.
        """
        nouns = pred["noun"] == gt_predictions["nouns"]
        boxes = ious.ravel() > self.iou_threshold

        map_box_noun = boxes & nouns
        map_box = boxes

        return np.vstack([map_box_noun, map_box]).T


class OverallMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Compute the different STA metrics based on mAP.

    This will compute the following metrics:
        - Box AP
        - Box + Noun AP
        - Box + Verb AP
        - Box + TTC AP
        - Box + Verb + TTC AP
        - Box + Noun mAP
        - Box + Noun + Verb mAP
        - Box + Noun + TTC mAP
        - Box + Noun + Verb + TTC mAP
    """

    def __init__(
        self, iou_threshold: float = 0.5, ttc_threshold: float = 0.25, top_k: int = 5, count_all_classes: bool = False
    ) -> None:
        """Construct the overall mAP metric.

        Args:
            iou_threshold: The IoU threshold to consider a prediction as a true positive.
            ttc_threshold: The TTC threshold to consider a prediction as a true positive.
            top_k: The K to be considered in the top-k criterion.
            count_all_classes: Whether to count all classes when computing the mAP.
        """
        super().__init__(num_aps=12, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.ttc_threshold = ttc_threshold
        self.names = [
            "Box AP",
            "Box + Noun AP",
            "Box + Verb AP",
            "Box + TTC AP",
            "Box + Noun + Verb AP",
            "Box + Noun + TTC AP",
            "Box + Verb + TTC AP",
            "Box + Noun + Verb + TTC AP",
            "Box + Noun mAP",
            "Box + Noun + Verb mAP",
            "Box + Noun + TTC mAP",
            "Box + Noun + Verb + TTC mAP",
        ]
        self.short_names = [
            "ap_box",
            "ap_box_noun",
            "ap_box_verb",
            "ap_box_ttc",
            "ap_box_noun_verb",
            "ap_box_noun_ttc",
            "ap_box_verb_ttc",
            "ap_box_noun_verb_ttc",
            "map_box_noun",
            "map_box_noun_verb",
            "map_box_noun_ttc",
            "map_box_noun_verb_ttc",
        ]

    def _map_classes(self, preds_labels: PredictionsFormat | LabelsFormat) -> npt.NDArray[np.float64]:
        """Associates each prediction to a class.

        Args:
            preds_labels: The predictions or labels containing class labels.

        Returns:
            An array of class indices associated with each prediction.
        """
        nouns = np.array(preds_labels["nouns"], dtype=np.float64)
        ones = np.ones(len(nouns), dtype=np.float64)

        return np.vstack(
            [
                ones,  # ap_box - do not average
                ones,  # ap_box_noun - do not average
                ones,  # ap_box_verb - do not average
                ones,  # ap_box_ttc - do not average
                ones,  # ap_box_noun_verb - do not average
                ones,  # ap_box_noun_ttc - do not average
                ones,  # ap_box_verb_ttc - do not average
                ones,  # ap_box_noun_verb_ttc - do not average
                nouns,  # map_box_noun - average over nouns
                nouns,  # map_box_noun_verb - average over nouns
                nouns,  # map_box_noun_ttc - average over nouns
                nouns,  # map_box_noun_verb_ttc - average over nouns
            ]
        ).T

    def _match(
        self, pred: Prediction, gt_predictions: LabelsFormat, ious: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Return matches of a given prediction to set of ground truth predictions.

        Args:
            pred: The prediction to match.
            gt_predictions: The ground truth label to match against.
            ious: A matrix of IoU values between the predictions and ground truth boxes.

        Returns:
            A matrix of shape (num_pred, num_aps), specifying possible matchings depending on the prediction and metric.
        """
        nouns = pred["noun"] == gt_predictions["nouns"]
        boxes = ious.ravel() > self.iou_threshold
        verbs = pred["verb"] == gt_predictions["verbs"]
        ttcs = (
            np.abs(np.array(pred["ttc"], dtype=np.float64) - np.array(gt_predictions["ttcs"], dtype=np.float64))
            <= self.ttc_threshold
        )

        tp_box_noun = boxes & nouns
        tp_box_verb = boxes & verbs
        tp_box_ttc = boxes & ttcs
        tp_box_noun_verb = boxes & nouns & verbs
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_verb_ttc = boxes & verbs & ttcs
        tp_box_noun_verb_ttc = boxes & nouns & verbs & ttcs

        return np.vstack(
            [
                boxes,  # ap_box
                tp_box_noun,  # ap_box_noun
                tp_box_verb,  # ap_box_verb
                tp_box_ttc,  # ap_box_ttc
                tp_box_noun_verb,  # ap_box_noun_verb
                tp_box_noun_ttc,  # ap_box_noun_ttc
                tp_box_verb_ttc,  # ap_box_verb_ttc
                tp_box_noun_verb_ttc,  # ap_box_noun_verb_ttc
                tp_box_noun,  # map_box_noun
                tp_box_noun_verb,  # map_box_noun_verb
                tp_box_noun_ttc,  # map_box_noun_ttc
                tp_box_noun_verb_ttc,  # map_box_noun_verb_ttc
            ]
        ).T


class STAMeanAveragePrecision(AbstractMeanAveragePrecision):
    """Compute the different STA metrics based on mAP.

    This will compute the following metrics:
        - Box + Noun mAP
        - Box + Noun + Verb mAP
        - Box + Noun + TTC mAP
        - Box + Noun + Verb + TTC mAP
    """

    def __init__(
        self, iou_threshold: float = 0.5, ttc_threshold: float = 0.25, top_k: int = 5, count_all_classes: bool = False
    ) -> None:
        """Construct the STA overall mAP metric.

        Args:
            iou_threshold: The IoU threshold to consider a prediction as a true positive.
            ttc_threshold: The TTC threshold to consider a prediction as a true positive.
            top_k: The K to be considered in the top-k criterion.
            count_all_classes: Whether to count all classes when computing the mAP.
        """
        super().__init__(num_aps=4, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.ttc_threshold = ttc_threshold
        self.names = [
            "Box + Noun mAP",
            "Box + Noun + Verb mAP",
            "Box + Noun + TTC mAP",
            "Box + Noun + Verb + TTC mAP",
        ]
        self.short_names = [
            "map_box_noun",
            "map_box_noun_verb",
            "map_box_noun_ttc",
            "map_box_noun_verb_ttc",
        ]

    def _map_classes(self, preds_labels: PredictionsFormat | LabelsFormat) -> npt.NDArray[np.float64]:
        """Associates each prediction to a class.

        Args:
            preds_labels: The predictions or labels containing class labels.

        Returns:
            An array of class indices associated with each prediction.
        """
        nouns = np.array(preds_labels["nouns"], dtype=np.float64)
        return np.vstack(
            [
                nouns,  # map_box_noun - average over nouns
                nouns,  # map_box_noun_verb - average over nouns
                nouns,  # map_box_noun_ttc - average over nouns
                nouns,  # map_box_noun_verb_ttc - average over nouns
            ]
        ).T

    def _match(
        self, pred: Prediction, gt_predictions: LabelsFormat, ious: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Return matches of a given prediction to set of ground truth predictions.

        Args:
            pred: The prediction to match.
            gt_predictions: The ground truth label to match against.
            ious: A matrix of IoU values between the predictions and ground truth boxes.

        Returns:
            A matrix of shape (num_pred, num_aps), specifying possible matchings depending on the prediction and metric.
        """
        nouns = pred["noun"] == gt_predictions["nouns"]
        boxes = ious.ravel() > self.iou_threshold
        verbs = pred["verb"] == gt_predictions["verbs"]
        ttcs = (
            np.abs(np.array(pred["ttc"], dtype=np.float64) - np.array(gt_predictions["ttcs"], dtype=np.float64))
            <= self.ttc_threshold
        )

        tp_box_noun = boxes & nouns
        tp_box_noun_verb = boxes & nouns & verbs
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_noun_verb_ttc = boxes & nouns & verbs & ttcs

        return np.vstack(
            [
                tp_box_noun,  # map_box_noun
                tp_box_noun_verb,  # map_box_noun_verb
                tp_box_noun_ttc,  # map_box_noun_ttc
                tp_box_noun_verb_ttc,  # map_box_noun_verb_ttc
            ]
        ).T
