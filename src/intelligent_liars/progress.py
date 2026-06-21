from __future__ import annotations

import os
import sys
import warnings
from typing import Any

from tqdm.rich import TqdmExperimentalWarning, tqdm


def progress_enabled() -> bool:
    """Return whether long-running CLI loops should render progress bars."""

    value = os.getenv("INTELLIGENT_LIARS_PROGRESS")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return sys.stderr.isatty()


def progress_bar(*, total: int, desc: str, unit: str, **kwargs: Any) -> tqdm:
    """Build one consistent Rich-backed tqdm progress bar."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmExperimentalWarning)
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=True,
            disable=not progress_enabled(),
            **kwargs,
        )
