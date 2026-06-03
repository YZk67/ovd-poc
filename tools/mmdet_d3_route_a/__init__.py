"""MMDetection plugins for the D3 Route-A GroundingDINO experiments."""

from .datasets import D3SubsetDODDataset  # noqa: F401
from .hooks import TrainableParamFreezeHook  # noqa: F401
from .models import GroundingDINOD3PromptWrapper, GroundingDINOTextQueryAdapter  # noqa: F401
