import gc
import numpy as np
from kneed import KneeLocator
from ripser import ripser

from entrap.constants import PERSISTENCE_ENTROPY_PERCENTILE_FALLBACK
from entrap.utils import optimize_memory


def compute_h0_diagram(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    n = len(X)
    if n < 2:
        return np.array([[0.0, 0.0]])

    try:
        dgms = ripser(X, distance_matrix=False, maxdim=0, metric=metric)['dgms']
        dgm_h0 = dgms[0]
        finite_mask = ~np.isinf(dgm_h0[:, 1])
        dgm_h0_finite = dgm_h0[finite_mask]

        if len(dgm_h0_finite) == 0:
            dgm_h0_finite = np.array([[0.0, 0.0]])

        return dgm_h0_finite

    except Exception:
        return np.array([[0.0, 0.0]])


def compute_persistence_entropy(diagram: np.ndarray) -> float:
    if len(diagram) == 0:
        return 0.0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 1e-12]

    if len(lifetimes) == 0:
        return 0.0

    L_total = np.sum(lifetimes)
    if L_total <= 1e-12:
        return 0.0

    p = lifetimes / L_total
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(entropy)


@optimize_memory
def compute_sequential_persistence_entropy(
    cluster_points: np.ndarray,
    candidates: np.ndarray,
    candidate_indices: np.ndarray,
    metric: str = 'euclidean'
):
    n_candidates = len(candidates)
    if n_candidates == 0:
        return np.array([]), []

    entropy_values = np.zeros(n_candidates, dtype=np.float64)
    current_data = cluster_points.copy()

    for i in range(n_candidates):
        current_data = np.vstack([current_data, candidates[i].reshape(1, -1)])
        diagram = compute_h0_diagram(current_data, metric=metric)
        entropy_values[i] = compute_persistence_entropy(diagram)

        if (i + 1) % 50 == 0:
            gc.collect()

    return entropy_values, candidate_indices


def detect_knee_with_kneed(
    entropy_values: np.ndarray,
    fallback_percentile: float = PERSISTENCE_ENTROPY_PERCENTILE_FALLBACK
) -> int:
    n = len(entropy_values)
    if n == 0:
        return 0
    if n == 1:
        return 1

    min_idx = int(np.argmin(entropy_values))
    auto_accept = min_idx + 1

    if min_idx == n - 1:
        return n

    post_entropy = entropy_values[min_idx:]
    if len(post_entropy) <= 2:
        return auto_accept

    try:
        kneedle = KneeLocator(
            np.arange(len(post_entropy)),
            post_entropy,
            curve='concave',
            direction='increasing',
            online=True
        )

        if kneedle.knee is not None:
            return max(auto_accept, min(min_idx + int(kneedle.knee), n))

        threshold = np.percentile(post_entropy, fallback_percentile)
        return max(
            auto_accept,
            min(min_idx + int(np.sum(post_entropy <= threshold)), n)
        )

    except Exception:
        return auto_accept
