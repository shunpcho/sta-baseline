"""Tests for sta_evaluate.py, comparing against EGO4D sta_metrics.py reference.

Reference: https://github.com/EGO4D/forecasting/blob/main/ego4d_forecasting/evaluation/sta_metrics.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sta_baseline.evaluation.sta_evaluate import (
    compute_iou,
    LabelsFormat,
    ObjectOnlyMeanAveragePrecision,
    OverallMeanAveragePrecision,
    PredictionsFormat,
    STAMeanAveragePrecision,
)

# ---------------------------------------------------------------------------
# Reference implementation (EGO4D/forecasting/ego4d_forecasting/evaluation/sta_metrics.py)
# Copied verbatim to serve as a ground-truth oracle for regression tests.
# ---------------------------------------------------------------------------


def _ref_compute_iou(preds: npt.NDArray[np.int64], gts: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    preds = np.expand_dims(preds, 1)
    gts = np.expand_dims(gts, 0)

    def area(boxes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        width = boxes[..., 2] - boxes[..., 0] + 1
        height = boxes[..., 3] - boxes[..., 1] + 1
        width[width < 0] = 0
        height[height < 0] = 0
        return width * height

    ixmin = np.maximum(gts[..., 0], preds[..., 0])
    iymin = np.maximum(gts[..., 1], preds[..., 1])
    ixmax = np.minimum(gts[..., 2], preds[..., 2])
    iymax = np.minimum(gts[..., 3], preds[..., 3])

    areas_preds = area(preds)
    areas_gts = area(gts)
    areas_intersections = area(np.stack([ixmin, iymin, ixmax, iymax], -1))
    return areas_intersections / (areas_preds + areas_gts - areas_intersections + 1e-11)


class _RefAbstractMAP(ABC):
    def __init__(
        self,
        num_aps: int,
        percentages: bool = True,
        count_all_classes: bool = True,
        top_k: int | None = None,
    ) -> None:
        self.true_positives: list[npt.NDArray[np.float64]] = []
        self.confidence_scores: list[npt.NDArray[np.float64]] = []
        self.predicted_classes: list[npt.NDArray[np.float64]] = []
        self.gt_classes: list[npt.NDArray[np.float64]] = []
        self.num_aps = num_aps
        self.percentages = percentages
        self.count_all_classes = count_all_classes
        self.K = top_k

    def add(self, preds: dict[str, Any], labels: dict[str, Any]) -> npt.NDArray[np.float64]:  # noqa: PLR0914
        matched: list[npt.NDArray[np.float64]] = []
        if len(preds) > 0:
            predicted_boxes = preds["boxes"]
            predicted_scores = preds["scores"]
            predicted_classes = self._map_classes(preds)
            true_positives = np.zeros((len(predicted_boxes), self.num_aps))

            if len(labels) > 0:
                gt_boxes = labels["boxes"]
                ious = _ref_compute_iou(predicted_boxes, gt_boxes)
                gt_matched = np.zeros((len(gt_boxes), self.num_aps))

                for i in predicted_scores.argsort()[::-1]:
                    overlaps = ious[i].reshape(-1, 1)
                    matchings = self._match({k: p[i] for k, p in preds.items()}, labels, overlaps)
                    overlaps = np.tile(overlaps, [1, matchings.shape[1]])
                    matchings[gt_matched == 1] = 0
                    overlaps[matchings == 0] = -1
                    jj = overlaps.argmax(0)
                    i_matchings = matchings[jj, range(len(jj))]
                    jj_matched = jj.copy()
                    jj_matched[~i_matchings] = -1
                    true_positives[i, i_matchings] = 1
                    gt_matched[jj, range(len(jj))] += i_matchings
                    matched.append(jj_matched)  # type: ignore[reportArgumentType]

                if self.K is not None and self.K > 1:
                    K = (self.K - 1) * len(labels["boxes"])  # noqa: N806
                    order = predicted_scores.argsort()[::-1]
                    sorted_tp = true_positives[order, :].astype(float)
                    sorted_fp = 1 - sorted_tp
                    sorted_tp[(sorted_fp.cumsum(0) <= K) & (sorted_fp == 1)] = np.nan
                    true_positives = sorted_tp
                    predicted_scores = predicted_scores[order]
                    predicted_classes = predicted_classes[order]

                self.gt_classes.append(self._map_classes(labels))

            self.true_positives.append(true_positives)
            self.confidence_scores.append(predicted_scores)
            self.predicted_classes.append(predicted_classes)

        if len(matched) > 0:
            return np.stack(matched, 0)
        return np.zeros((0, self.num_aps))

    def _map_classes(self, preds: dict[str, Any]) -> npt.NDArray[np.float64]:
        return np.vstack([preds["nouns"]] * self.num_aps).T

    def _compute_prec_rec(
        self,
        true_positives: npt.NDArray[np.float64],
        confidence_scores: npt.NDArray[np.float64],
        num_gt: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        tps = true_positives[confidence_scores.argsort()[::-1]]
        tp = tps.cumsum()
        fp = (1 - tps).cumsum()
        prec = self._safe_division(tp, tp + fp)
        rec = self._safe_division(tp, num_gt)
        return np.asarray(prec, dtype=float), np.asarray(rec, dtype=float)

    def _safe_division(
        self,
        a: npt.NDArray[np.float64] | float,
        b: npt.NDArray[np.float64] | float,
    ) -> npt.NDArray[np.float64] | float:
        a_array = isinstance(a, np.ndarray)
        b_arr = isinstance(b, np.ndarray)
        if not a_array and not b_arr and b == 0:
            return 0.0
        if b_arr and not a_array:
            a = np.array([a] * len(b))
        if not b_arr and a_array:
            b = np.array([b] * np.size(a))
        b = np.asarray(b, dtype=float)
        a = np.asarray(a, dtype=float)
        zeroden = b == 0
        b[zeroden] = 1
        a[zeroden] = 0
        return a / b

    def _compute_ap(self, prec: npt.NDArray[np.float64], rec: npt.NDArray[np.float64]) -> float:
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.concatenate(([0], prec, [0]))
        for i in range(len(mpre) - 2, 0, -1):
            mpre[i] = np.max((mpre[i], mpre[i + 1]))
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        return float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))

    def evaluate(self, measure: str = "AP") -> tuple[float, ...] | float:  # noqa: PLR0914, C901
        metrics: list[float] = []
        gt_classes = np.concatenate(self.gt_classes)
        predicted_classes = np.concatenate(self.predicted_classes)
        true_positives = np.concatenate(self.true_positives)
        confidence_scores = np.concatenate(self.confidence_scores)

        for i in range(self.num_aps):
            measures: list[float] = []
            gt_classes_i = gt_classes[:, i]
            predicted_classes_i = predicted_classes[:, i]
            true_positives_i = true_positives[:, i]

            if self.count_all_classes:
                classes = np.unique(np.concatenate([gt_classes_i, predicted_classes_i]))
            else:
                classes = np.unique(gt_classes_i)

            for c in classes:
                tp = true_positives_i[predicted_classes_i == c]
                cs = confidence_scores[predicted_classes_i == c]
                ngt = int(np.sum(gt_classes_i == c))  # type: ignore[reportUnknownArgumentType]

                if len(tp) > 0:
                    valid = ~np.isnan(tp)
                    tp, cs = tp[valid], cs[valid]

                this_measure: float = 0.0
                if len(tp) > 0 and ngt > 0:
                    prec, rec = self._compute_prec_rec(tp, cs, ngt)
                    if measure == "AP":
                        this_measure: float = self._compute_ap(prec, rec)
                    elif measure == "MR":
                        this_measure = float(np.max(rec))
                    if self.percentages:
                        this_measure *= 100
                    measures.append(this_measure)
                elif not (len(tp) == 0 and ngt == 0):
                    measures.append(0.0)

            metrics.append(float(np.mean(measures)))

        values = list(metrics)
        if len(values) == 1:
            return values[0]
        return tuple(values)

    @abstractmethod
    def _match(
        self, pred: dict[str, Any], gt_predictions: dict[str, Any], ious: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        pass


class _RefSTAMAP(_RefAbstractMAP):
    """Reference STAMeanAveragePrecision from EGO4D sta_metrics.py."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        ttc_threshold: float = 0.25,
        top_k: int = 5,
        count_all_classes: bool = False,
    ) -> None:
        super().__init__(4, top_k=top_k, count_all_classes=count_all_classes)
        self.iou_threshold = iou_threshold
        self.tti_threshold = ttc_threshold

    def _map_classes(self, preds: dict[str, Any]) -> npt.NDArray[np.float64]:
        nouns = preds["nouns"]
        return np.vstack([nouns] * 4).T

    def _match(
        self, pred: dict[str, Any], gt_predictions: dict[str, Any], ious: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        nouns = pred["nouns"] == gt_predictions["nouns"]
        boxes = ious.ravel() > self.iou_threshold
        verbs = pred["verbs"] == gt_predictions["verbs"]
        ttcs = np.abs(pred["ttcs"] - gt_predictions["ttcs"]) <= self.tti_threshold

        tp_box_noun = boxes & nouns
        tp_box_noun_verb = boxes & verbs & nouns
        tp_box_noun_ttc = boxes & nouns & ttcs
        tp_box_noun_verb_ttc = boxes & verbs & nouns & ttcs

        return np.vstack([tp_box_noun, tp_box_noun_verb, tp_box_noun_ttc, tp_box_noun_verb_ttc]).T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_preds(
    boxes: list[list[int]],
    scores: list[float],
    nouns: list[int],
    verbs: list[int],
    ttcs: list[float],
) -> PredictionsFormat:
    return PredictionsFormat(
        boxes=np.array(boxes, dtype=np.uint16),
        scores=np.array(scores, dtype=np.float32),
        nouns=np.array(nouns, dtype=np.uint16),
        verbs=np.array(verbs, dtype=np.uint16),
        ttcs=np.array(ttcs, dtype=np.float32),
    )


def _make_labels(
    boxes: list[list[int]],
    nouns: list[int],
    verbs: list[int],
    ttcs: list[float],
) -> LabelsFormat:
    return LabelsFormat(
        boxes=np.array(boxes, dtype=np.uint16),
        nouns=np.array(nouns, dtype=np.uint16),
        verbs=np.array(verbs, dtype=np.uint16),
        ttcs=np.array(ttcs, dtype=np.float32),
    )


def _ref_preds(
    boxes: list[list[int]],
    scores: list[float],
    nouns: list[int],
    verbs: list[int],
    ttcs: list[float],
) -> dict[str, Any]:
    return {
        "boxes": np.array(boxes),
        "scores": np.array(scores),
        "nouns": np.array(nouns),
        "verbs": np.array(verbs),
        "ttcs": np.array(ttcs, dtype=np.float32),
    }


def _ref_labels(
    boxes: list[list[int]],
    nouns: list[int],
    verbs: list[int],
    ttcs: list[float],
) -> dict[str, Any]:
    return {
        "boxes": np.array(boxes),
        "nouns": np.array(nouns),
        "verbs": np.array(verbs),
        "ttcs": np.array(ttcs, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Tests: compute_iou
# ---------------------------------------------------------------------------


class TestComputeIou:
    def test_identical_boxes_iou_is_one(self) -> None:
        iou = compute_iou([[0, 0, 9, 9]], [[0, 0, 9, 9]])
        assert iou.shape == (1, 1)
        np.testing.assert_allclose(iou, [[1.0]], atol=1e-5)

    def test_non_overlapping_iou_is_zero(self) -> None:
        iou = compute_iou([[0, 0, 4, 4]], [[10, 10, 14, 14]])
        assert iou.shape == (1, 1)
        assert float(iou[0, 0]) == pytest.approx(0.0, abs=1e-5)

    def test_partial_overlap_known_value(self) -> None:
        # [0,0,9,9] area=100; [5,0,14,9] area=100
        # intersection [5,0,9,9]: width=5, height=10 → area=50
        # iou = 50 / (100 + 100 - 50) = 1/3
        iou = compute_iou([[0, 0, 9, 9]], [[5, 0, 14, 9]])
        np.testing.assert_allclose(iou, [[1.0 / 3.0]], atol=1e-5)

    def test_output_shape_broadcasting(self) -> None:
        preds = [[0, 0, 9, 9], [10, 10, 19, 19], [20, 20, 29, 29]]
        gts = [[0, 0, 9, 9], [5, 5, 14, 14]]
        iou = compute_iou(preds, gts)
        assert iou.shape == (3, 2)

    def test_output_dtype_is_float32(self) -> None:
        iou = compute_iou([[0, 0, 9, 9]], [[0, 0, 9, 9]])
        assert iou.dtype == np.float32

    def test_iou_values_are_in_zero_one_range(self) -> None:
        preds = [[0, 0, 9, 9], [100, 100, 200, 200]]
        gts = [[5, 5, 14, 14], [0, 0, 9, 9], [100, 100, 200, 200]]
        iou = compute_iou(preds, gts)
        assert np.all(iou >= 0.0)
        assert np.all(iou <= 1.0 + 1e-6)

    def test_matches_reference_implementation(self) -> None:
        preds = np.array([[245, 128, 589, 683], [425, 68, 592, 128], [120, 200, 180, 260]])
        gts = np.array([[195, 322, 625, 800], [150, 300, 425, 689], [121, 201, 181, 261]])
        our_iou = compute_iou(preds, gts)
        ref_iou = _ref_compute_iou(preds.copy(), gts.copy())
        np.testing.assert_allclose(our_iou, ref_iou.astype(np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: AbstractMeanAveragePrecision internal methods
# (accessed via the concrete STAMeanAveragePrecision)
# ---------------------------------------------------------------------------


class TestAbstractMapInternals:
    def setup_method(self) -> None:
        self.metric = STAMeanAveragePrecision()

    def test_compute_precision_recall_all_tp(self) -> None:
        tp = np.array([1.0, 1.0, 1.0])
        scores = np.array([0.9, 0.8, 0.7])
        prec, rec = self.metric._compute_precision_recall(tp, scores, 3)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        np.testing.assert_allclose(prec, [1.0, 1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(rec, [1 / 3, 2 / 3, 1.0], atol=1e-6)

    def test_compute_precision_recall_all_fp(self) -> None:
        tp = np.array([0.0, 0.0, 0.0])
        scores = np.array([0.9, 0.8, 0.7])
        prec, rec = self.metric._compute_precision_recall(tp, scores, 3)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        np.testing.assert_allclose(prec, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(rec, [0.0, 0.0, 0.0], atol=1e-6)

    def test_safe_divide_zero_denominator_returns_zero(self) -> None:
        result = self.metric._safe_divide(np.array([1.0, 2.0]), np.array([0.0, 4.0]))  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        np.testing.assert_allclose(result, [0.0, 0.5], atol=1e-6)

    def test_safe_divide_normal_case(self) -> None:
        result = self.metric._safe_divide(np.array([3.0, 6.0]), np.array([3.0, 2.0]))  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        np.testing.assert_allclose(result, [1.0, 3.0], atol=1e-6)

    def test_compute_average_precision_perfect(self) -> None:
        prec = np.array([1.0, 1.0, 1.0])
        rec = np.array([1 / 3, 2 / 3, 1.0])
        ap = self.metric._compute_average_precision(prec, rec)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        assert float(ap) == pytest.approx(1.0, abs=1e-5)

    def test_compute_average_precision_zero(self) -> None:
        prec = np.array([0.0, 0.0])
        rec = np.array([0.0, 0.0])
        ap = self.metric._compute_average_precision(prec, rec)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        assert float(ap) == pytest.approx(0.0, abs=1e-5)

    def test_compute_max_recall_returns_max(self) -> None:
        rec = np.array([0.2, 0.5, 0.8, 0.6])
        assert float(self.metric._compute_max_recall(rec)) == pytest.approx(0.8, abs=1e-6)  # noqa: SLF001 # type: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Tests: ObjectOnlyMeanAveragePrecision
# ---------------------------------------------------------------------------


class TestObjectOnlyMAP:
    def test_map_classes_shape(self) -> None:
        preds = _make_preds([[0, 0, 9, 9], [10, 10, 19, 19]], [0.9, 0.8], [1, 2], [3, 4], [1.0, 2.0])
        metric = ObjectOnlyMeanAveragePrecision()
        classes = metric._map_classes(preds)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        assert classes.shape == (2, 2)

    def test_map_classes_first_column_is_nouns(self) -> None:
        preds = _make_preds([[0, 0, 9, 9], [10, 10, 19, 19]], [0.9, 0.8], [5, 7], [3, 4], [1.0, 2.0])
        metric = ObjectOnlyMeanAveragePrecision()
        classes = metric._map_classes(preds)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        np.testing.assert_array_equal(classes[:, 0], [5, 7])

    def test_map_classes_second_column_is_ones(self) -> None:
        preds = _make_preds([[0, 0, 9, 9], [10, 10, 19, 19]], [0.9, 0.8], [5, 7], [3, 4], [1.0, 2.0])
        metric = ObjectOnlyMeanAveragePrecision()
        classes = metric._map_classes(preds)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        np.testing.assert_array_equal(classes[:, 1], [1, 1])

    def test_perfect_prediction_returns_100(self) -> None:
        preds = _make_preds([[10, 20, 50, 80]], [0.9], [3], [7], [1.0])
        labels = _make_labels([[10, 20, 50, 80]], [3], [7], [1.0])
        metric = ObjectOnlyMeanAveragePrecision()
        metric.add(preds, labels)
        result = metric.evaluate()
        assert result == (pytest.approx(100.0, abs=1e-3), pytest.approx(100.0, abs=1e-3))

    def test_no_overlap_returns_zero(self) -> None:
        preds = _make_preds([[200, 200, 300, 300]], [0.9], [3], [7], [1.0])
        labels = _make_labels([[0, 0, 9, 9]], [3], [7], [1.0])
        metric = ObjectOnlyMeanAveragePrecision()
        metric.add(preds, labels)
        result = metric.evaluate()
        assert result == (pytest.approx(0.0, abs=1e-3), pytest.approx(0.0, abs=1e-3))

    def test_wrong_noun_gives_zero_map_box_noun(self) -> None:
        # Box overlaps perfectly but noun is wrong
        preds = _make_preds([[10, 20, 50, 80]], [0.9], [99], [7], [1.0])
        labels = _make_labels([[10, 20, 50, 80]], [3], [7], [1.0])
        metric = ObjectOnlyMeanAveragePrecision()
        metric.add(preds, labels)
        map_box_noun, ap_box = metric.evaluate()
        assert map_box_noun == pytest.approx(0.0, abs=1e-3)
        assert ap_box == pytest.approx(100.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Tests: STAMeanAveragePrecision
# ---------------------------------------------------------------------------


class TestSTAMAP:
    def test_perfect_prediction_all_100(self) -> None:
        preds = _make_preds([[10, 20, 50, 80]], [0.9], [3], [7], [1.0])
        labels = _make_labels([[10, 20, 50, 80]], [3], [7], [1.0])
        metric = STAMeanAveragePrecision()
        metric.add(preds, labels)
        assert metric.evaluate() == (
            pytest.approx(100.0, abs=1e-3),
            pytest.approx(100.0, abs=1e-3),
            pytest.approx(100.0, abs=1e-3),
            pytest.approx(100.0, abs=1e-3),
        )

    def test_no_overlap_all_zero(self) -> None:
        preds = _make_preds([[200, 200, 300, 300]], [0.9], [3], [7], [1.0])
        labels = _make_labels([[0, 0, 9, 9]], [3], [7], [1.0])
        metric = STAMeanAveragePrecision()
        metric.add(preds, labels)
        assert metric.evaluate() == (
            pytest.approx(0.0, abs=1e-3),
            pytest.approx(0.0, abs=1e-3),
            pytest.approx(0.0, abs=1e-3),
            pytest.approx(0.0, abs=1e-3),
        )

    def test_wrong_verb_drops_verb_metrics(self) -> None:
        # Box and noun match, but verb does not → map_box_noun_verb and map_box_noun_verb_ttc are 0
        preds = _make_preds([[10, 20, 50, 80]], [0.9], [3], [99], [1.0])
        labels = _make_labels([[10, 20, 50, 80]], [3], [7], [1.0])
        metric = STAMeanAveragePrecision()
        metric.add(preds, labels)
        map_noun, map_noun_verb, map_noun_ttc, map_noun_verb_ttc = metric.evaluate()
        assert map_noun == pytest.approx(100.0, abs=1e-3)
        assert map_noun_verb == pytest.approx(0.0, abs=1e-3)
        assert map_noun_ttc == pytest.approx(100.0, abs=1e-3)
        assert map_noun_verb_ttc == pytest.approx(0.0, abs=1e-3)

    def test_wrong_ttc_drops_ttc_metrics(self) -> None:
        # Box, noun, verb match, but ttc is far off
        preds = _make_preds([[10, 20, 50, 80]], [0.9], [3], [7], [5.0])
        labels = _make_labels([[10, 20, 50, 80]], [3], [7], [1.0])
        metric = STAMeanAveragePrecision(ttc_threshold=0.25)
        metric.add(preds, labels)
        map_noun, map_noun_verb, map_noun_ttc, map_noun_verb_ttc = metric.evaluate()
        assert map_noun == pytest.approx(100.0, abs=1e-3)
        assert map_noun_verb == pytest.approx(100.0, abs=1e-3)
        assert map_noun_ttc == pytest.approx(0.0, abs=1e-3)
        assert map_noun_verb_ttc == pytest.approx(0.0, abs=1e-3)

    def test_evaluate_returns_four_element_tuple(self) -> None:
        preds = _make_preds([[0, 0, 9, 9]], [0.5], [1], [1], [1.0])
        labels = _make_labels([[0, 0, 9, 9]], [1], [1], [1.0])
        metric = STAMeanAveragePrecision()
        metric.add(preds, labels)
        result = metric.evaluate()
        assert len(result) == 4

    def test_multiple_images_accumulated(self) -> None:
        # Two perfect predictions on two separate calls
        preds1 = _make_preds([[0, 0, 9, 9]], [0.9], [1], [2], [0.5])
        labels1 = _make_labels([[0, 0, 9, 9]], [1], [2], [0.5])
        preds2 = _make_preds([[50, 50, 99, 99]], [0.8], [3], [4], [1.0])
        labels2 = _make_labels([[50, 50, 99, 99]], [3], [4], [1.0])

        metric = STAMeanAveragePrecision()
        metric.add(preds1, labels1)
        metric.add(preds2, labels2)
        result = metric.evaluate()
        np.testing.assert_allclose(result, (100.0, 100.0, 100.0, 100.0), atol=1e-3)

    def test_map_noun_averages_across_noun_classes(self) -> None:
        # Two noun classes: class 1 has a perfect prediction, class 2 has no prediction
        # map_box_noun averages over noun classes → should be 50% (100+0)/2
        preds = _make_preds([[0, 0, 9, 9]], [0.9], [1], [1], [1.0])
        labels = _make_labels([[0, 0, 9, 9], [50, 50, 99, 99]], [1, 2], [1, 2], [1.0, 1.0])
        metric = STAMeanAveragePrecision()
        metric.add(preds, labels)
        map_noun, *_ = metric.evaluate()
        assert map_noun == pytest.approx(50.0, abs=1e-3)

    def test_top_k_discards_only_false_positives(self) -> None:
        # 3 predictions for 1 GT box (noun=1):
        #   FP(score=0.9, no overlap), TP(score=0.8, perfect overlap), FP(score=0.7, no overlap)
        # top_k=2 → K=(2-1)*1=1 highest-scoring FP is NaN'd → the TP remains
        preds = _make_preds(
            [[200, 200, 300, 300], [0, 0, 9, 9], [400, 400, 500, 500]],
            [0.9, 0.8, 0.7],
            [1, 1, 1],
            [1, 1, 1],
            [1.0, 1.0, 1.0],
        )
        labels = _make_labels([[0, 0, 9, 9]], [1], [1], [1.0])

        metric_topk = STAMeanAveragePrecision(top_k=2)
        metric_topk.add(preds, labels)
        result_topk = metric_topk.evaluate()

        # With top_k=2, the highest-score FP is discarded, TP is preserved → AP > 0
        assert result_topk[0] > 0.0

    def test_get_names_returns_four_strings(self) -> None:
        metric = STAMeanAveragePrecision()
        assert len(metric.get_names()) == 4
        assert len(metric.get_short_names()) == 4


# ---------------------------------------------------------------------------
# Tests: OverallMeanAveragePrecision
# ---------------------------------------------------------------------------


class TestOverallMAP:
    def test_map_classes_shape_is_n_by_12(self) -> None:
        preds = _make_preds(
            [[0, 0, 9, 9], [10, 10, 19, 19], [20, 20, 29, 29]], [0.9, 0.8, 0.7], [1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0]
        )
        metric = OverallMeanAveragePrecision()
        classes = metric._map_classes(preds)  # noqa: SLF001 # type: ignore[reportPrivateUsage]
        assert classes.shape == (3, 12)

    def test_evaluate_returns_12_element_tuple(self) -> None:
        preds = _make_preds([[0, 0, 9, 9]], [0.9], [1], [1], [1.0])
        labels = _make_labels([[0, 0, 9, 9]], [1], [1], [1.0])
        metric = OverallMeanAveragePrecision()
        metric.add(preds, labels)
        result = metric.evaluate()
        assert len(result) == 12

    def test_perfect_prediction_all_100(self) -> None:
        preds = _make_preds([[0, 0, 9, 9]], [0.9], [1], [1], [1.0])
        labels = _make_labels([[0, 0, 9, 9]], [1], [1], [1.0])
        metric = OverallMeanAveragePrecision()
        metric.add(preds, labels)
        result = metric.evaluate()
        np.testing.assert_allclose(result, [100.0] * 12, atol=1e-3)

    def test_get_short_names_count(self) -> None:
        metric = OverallMeanAveragePrecision()
        assert len(metric.get_short_names()) == 12
        assert "ap_box" in metric.get_short_names()
        assert "map_box_noun_verb_ttc" in metric.get_short_names()


# ---------------------------------------------------------------------------
# Regression tests: compare against EGO4D reference implementation
# ---------------------------------------------------------------------------


class TestRegressionVsReference:
    """Verify STAMeanAveragePrecision matches the EGO4D reference on identical inputs."""

    def __init__(self) -> None:
        self.BOXES_PREDS = [[245, 128, 589, 683], [425, 68, 592, 128], [120, 200, 180, 260], [150, 150, 250, 250]]
        self.SCORES = [0.8, 0.4, 0.9, 0.1]
        self.NOUNS = [3, 5, 7, 9]
        self.VERBS = [8, 11, 6, 10]
        self.TTCS = [1.25, 1.8, 2.0, 2.5]
        self.BOXES_LABELS = [[195, 322, 625, 800], [150, 300, 425, 689], [121, 201, 181, 261], [100, 100, 200, 200]]
        self.NOUNS_GT = [9, 5, 7, 1]
        self.VERBS_GT = [3, 11, 6, 2]
        self.TTCS_GT = [0.25, 1.25, 2.0, 3.0]

    def _make_our(self, top_k: int = 5) -> STAMeanAveragePrecision:
        preds = _make_preds(self.BOXES_PREDS, self.SCORES, self.NOUNS, self.VERBS, self.TTCS)
        labels = _make_labels(self.BOXES_LABELS, self.NOUNS_GT, self.VERBS_GT, self.TTCS_GT)
        metric = STAMeanAveragePrecision(top_k=top_k)
        metric.add(preds, labels)
        return metric

    def _make_ref(self, top_k: int = 5) -> _RefSTAMAP:
        preds = _ref_preds(self.BOXES_PREDS, self.SCORES, self.NOUNS, self.VERBS, self.TTCS)
        labels = _ref_labels(self.BOXES_LABELS, self.NOUNS_GT, self.VERBS_GT, self.TTCS_GT)
        metric = _RefSTAMAP(top_k=top_k)
        metric.add(preds, labels)
        return metric

    def test_single_image_ap_matches_reference(self) -> None:
        our = self._make_our()
        ref = self._make_ref()
        np.testing.assert_allclose(our.evaluate(), ref.evaluate(), atol=1e-4)

    def test_single_image_mr_matches_reference(self) -> None:
        our = self._make_our()
        ref = self._make_ref()
        np.testing.assert_allclose(our.evaluate("MR"), ref.evaluate("MR"), atol=1e-4)

    def test_top_k_1_matches_reference(self) -> None:
        our = self._make_our(top_k=1)
        ref = self._make_ref(top_k=1)
        np.testing.assert_allclose(our.evaluate(), ref.evaluate(), atol=1e-4)

    def test_top_k_2_matches_reference(self) -> None:
        our = self._make_our(top_k=2)
        ref = self._make_ref(top_k=2)
        np.testing.assert_allclose(our.evaluate(), ref.evaluate(), atol=1e-4)

    def test_multi_image_matches_reference(self) -> None:
        preds1 = _make_preds(self.BOXES_PREDS[:2], self.SCORES[:2], self.NOUNS[:2], self.VERBS[:2], self.TTCS[:2])
        labels1 = _make_labels(self.BOXES_LABELS[:2], self.NOUNS_GT[:2], self.VERBS_GT[:2], self.TTCS_GT[:2])
        preds2 = _make_preds(self.BOXES_PREDS[2:], self.SCORES[2:], self.NOUNS[2:], self.VERBS[2:], self.TTCS[2:])
        labels2 = _make_labels(self.BOXES_LABELS[2:], self.NOUNS_GT[2:], self.VERBS_GT[2:], self.TTCS_GT[2:])

        our = STAMeanAveragePrecision()
        our.add(preds1, labels1)
        our.add(preds2, labels2)

        ref_p1 = _ref_preds(self.BOXES_PREDS[:2], self.SCORES[:2], self.NOUNS[:2], self.VERBS[:2], self.TTCS[:2])
        ref_l1 = _ref_labels(self.BOXES_LABELS[:2], self.NOUNS_GT[:2], self.VERBS_GT[:2], self.TTCS_GT[:2])
        ref_p2 = _ref_preds(self.BOXES_PREDS[2:], self.SCORES[2:], self.NOUNS[2:], self.VERBS[2:], self.TTCS[2:])
        ref_l2 = _ref_labels(self.BOXES_LABELS[2:], self.NOUNS_GT[2:], self.VERBS_GT[2:], self.TTCS_GT[2:])

        ref = _RefSTAMAP()
        ref.add(ref_p1, ref_l1)
        ref.add(ref_p2, ref_l2)

        np.testing.assert_allclose(our.evaluate(), ref.evaluate(), atol=1e-4)
