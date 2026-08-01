from sta_baseline.models.build import MODEL_REGISTRY, build_model  # noqa
from sta_baseline.models.video_model_builder import ResNet, SlowFast  # noqa

from sta_baseline.models.sta_models import (
    ShortTermAnticipationResNet as ShortTermAnticipationResNet,
    ShortTermAnticipationSlowFast as ShortTermAnticipationSlowFast,
)
