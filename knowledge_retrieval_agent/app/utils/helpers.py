"""General helper utilities used by multiple modules."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


def min_max_normalize(value: float, min_value: float, max_value: float) -> float:
    """Normalize a value into the [0, 1] range."""

    if max_value <= min_value:
        return 0.0
    return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))


@contextmanager
def measure_time() -> Generator[dict[str, float], None, None]:
    """Measure elapsed time in milliseconds inside a context manager."""

    state = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        end = time.perf_counter()
        state["elapsed_ms"] = (end - start) * 1000
