"""YeetLLM package."""

import os
from importlib.metadata import PackageNotFoundError, version

__version__ = os.environ.get("YEETLLM_IMAGE_VERSION", "")
if not __version__:
    try:
        __version__ = version("yeetllm")
    except PackageNotFoundError:  # pragma: no cover - source checkout
        __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
