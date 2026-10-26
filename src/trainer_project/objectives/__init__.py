from .base import Objective, ObjectiveResult, global_masked_mean, target_mask
from .classification import (
    ArcFaceObjective,
    FocalLoss,
    LDAMLoss,
    LogitAdjustedLoss,
    MaskedCrossEntropy,
    SeesawLoss,
    TripletLoss,
)
from .composite import CompositeObjective
from .inference import objective_inference_outputs
from .segmentation import SegmentationObjective
from .ssl import BYOLObjective, MAEObjective

# Explicit aliases used by common recipe terminology.
SubcenterArcFaceObjective = ArcFaceObjective
AAMLoss = ArcFaceObjective
MaskedSegmentationLoss = SegmentationObjective

__all__ = [
    "Objective",
    "ObjectiveResult",
    "global_masked_mean",
    "target_mask",
    "ArcFaceObjective",
    "FocalLoss",
    "LDAMLoss",
    "LogitAdjustedLoss",
    "MaskedCrossEntropy",
    "SeesawLoss",
    "TripletLoss",
    "CompositeObjective",
    "SegmentationObjective",
    "BYOLObjective",
    "MAEObjective",
    "SubcenterArcFaceObjective",
    "AAMLoss",
    "MaskedSegmentationLoss",
    "objective_inference_outputs",
]
