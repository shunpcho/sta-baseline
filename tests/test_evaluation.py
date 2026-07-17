import numpy as np

from sta_baseline.evaluation.sta_evaluate import compute_iou, LabelsFormat, PredictionsFormat

preds = PredictionsFormat(
    boxes=np.array(
        [[245, 128, 589, 683], [425, 68, 592, 128], [120, 200, 180, 260], [150, 150, 250, 250]], dtype=np.uint16
    ),
    scores=np.array([0.8, 0.4, 0.9, 0.1], dtype=np.float32),
    nouns=np.array([3, 5, 7, 9], dtype=np.uint16),
    verbs=np.array([8, 11, 6, 10], dtype=np.uint16),
    ttcs=np.array([1.25, 1.8, 2.0, 2.5], dtype=np.float32),
)

labels = LabelsFormat(
    boxes=np.array(
        [[195, 322, 625, 800], [150, 300, 425, 689], [121, 201, 181, 261], [100, 100, 200, 200]], dtype=np.uint16
    ),
    nouns=np.array([9, 5, 7, 1], dtype=np.uint16),
    verbs=np.array([3, 11, 6, 2], dtype=np.uint16),
    ttcs=np.array([0.25, 1.25, 2.0, 3.0], dtype=np.float32),
)

ious = compute_iou(preds["boxes"], labels["boxes"])
