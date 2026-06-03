import gc
from functools import wraps
from typing import Callable

from entrap.constants import SUPPORTED_METRICS


def optimize_memory(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper


def validate_metric(metric):
    if callable(metric):
        return metric
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Metric '{metric}' not supported. "
            f"Supported metrics: {SUPPORTED_METRICS}"
        )
    return metric
