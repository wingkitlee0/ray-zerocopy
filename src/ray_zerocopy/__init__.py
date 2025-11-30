import importlib.metadata

from .model_wrappers import ModelWrapper
from .wrappers import JITModelWrapper

__version__ = importlib.metadata.version("ray-zerocopy")


__all__ = [
    "__version__",
    "ModelWrapper",
    "JITModelWrapper",
]
