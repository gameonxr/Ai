from __future__ import annotations

from copy import deepcopy
from typing import Any
import math

import numpy as np


def _map_numeric(value: Any, fn):
    if isinstance(value, dict):
        return {key: _map_numeric(item, fn) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_map_numeric(item, fn) for item in value]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return fn(float(value))
    return deepcopy(value)


class GaussianNoise:
    """Add seeded Gaussian noise to numeric sensor values."""

    def __init__(self, standard_deviation: float = 0.0, seed: int | None = None):
        if isinstance(standard_deviation, bool) or not isinstance(standard_deviation, (int, float, np.integer, np.floating)) or not math.isfinite(float(standard_deviation)) or float(standard_deviation) < 0:
            raise ValueError("standard_deviation must be a finite non-negative number")
        self._validate_seed(seed)
        self.standard_deviation = float(standard_deviation)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self._validate_seed(seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_seed(seed: int | None) -> None:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")

    def apply(self, value: Any):
        return _map_numeric(value, lambda item: item + float(self.rng.normal(0.0, self.standard_deviation))) if self.standard_deviation else deepcopy(value)


class LowPassFilter:
    """First-order exponential filter for nested numeric readings."""

    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)) or not math.isfinite(float(alpha)) or not 0 < float(alpha) <= 1:
            raise ValueError("alpha must be a finite number in (0, 1]")
        self.alpha = float(alpha)
        self.previous = None

    def reset(self) -> None:
        self.previous = None

    def apply(self, value: Any):
        if self.previous is None:
            self.previous = deepcopy(value)
            return deepcopy(value)
        current = self.previous
        if isinstance(value, dict):
            result = {key: self.apply_pair(current.get(key), item) for key, item in value.items()}
        else:
            result = self.apply_pair(current, value)
        self.previous = deepcopy(result)
        return result

    def apply_pair(self, previous: Any, value: Any):
        if isinstance(value, dict):
            return {key: self.apply_pair((previous or {}).get(key), item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            previous = previous if isinstance(previous, (list, tuple, np.ndarray)) else [0.0] * len(value)
            return [self.apply_pair(old, new) for old, new in zip(previous, value)]
        if isinstance(value, (int, float, np.integer, np.floating)) and isinstance(previous, (int, float, np.integer, np.floating)):
            return self.alpha * float(value) + (1.0 - self.alpha) * float(previous)
        return deepcopy(value)
