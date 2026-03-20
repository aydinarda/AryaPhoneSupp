from __future__ import annotations

import warnings

warnings.warn(
    "app.mincost_agent is deprecated; use app.optimization_controller instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .optimization_controller import *  # noqa: F401,F403