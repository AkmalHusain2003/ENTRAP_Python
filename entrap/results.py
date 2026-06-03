import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from entrap.tracker import EntropyProgressTracker


@dataclass
class ENTRAP_Results:
    labels: np.ndarray
    probabilities: np.ndarray
    noise_rescued: int
    execution_time: float
    n_clusters: int
    cluster_stats: Dict
    hyperparameters: Optional[Dict] = None
    tracker: Optional[EntropyProgressTracker] = None
