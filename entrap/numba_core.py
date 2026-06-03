import numpy as np
from numba import jit, njit, prange


@jit(nopython=True, fastmath=True)
def compute_cov_from_rows(neighbor_distances: np.ndarray) -> np.ndarray:
    n_points, m = neighbor_distances.shape
    covs = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        row_sum = 0.0
        for j in range(m):
            row_sum += neighbor_distances[i, j]
        mean = row_sum / m

        if mean > 1e-12:
            variance_sum = 0.0
            for j in range(m):
                diff = neighbor_distances[i, j] - mean
                variance_sum += diff * diff
            std = np.sqrt(variance_sum / m)
            covs[i] = std / mean
        else:
            covs[i] = np.inf

    return covs


@njit(fastmath=True)
def compute_cluster_mean(cluster_points: np.ndarray) -> np.ndarray:
    n, d = cluster_points.shape
    mean = np.zeros(d, dtype=np.float64)

    for i in range(n):
        for j in range(d):
            mean[j] += cluster_points[i, j]

    for j in range(d):
        mean[j] /= n

    return mean


@njit(fastmath=True, parallel=True)
def compute_cluster_covariance(
    cluster_points: np.ndarray,
    mean: np.ndarray,
    ridge_epsilon: float
) -> np.ndarray:
    n, d = cluster_points.shape
    cov = np.zeros((d, d), dtype=np.float64)

    for i in prange(n):
        diff = cluster_points[i, :] - mean
        for j in range(d):
            for k in range(d):
                cov[j, k] += diff[j] * diff[k]

    for j in range(d):
        for k in range(d):
            cov[j, k] /= n
            if j == k:
                cov[j, k] += ridge_epsilon

    return cov


@njit(fastmath=True)
def compute_mahalanobis_sq(diff: np.ndarray, Sigma_inv: np.ndarray) -> float:
    result = 0.0
    for i in range(len(diff)):
        temp = 0.0
        for j in range(len(diff)):
            temp += diff[j] * Sigma_inv[j, i]
        result += diff[i] * temp
    return result


@njit(fastmath=True)
def logistic_mapping(
    cov_value: float,
    cov_10: float,
    cov_50: float,
    cov_90: float,
    q_min: float,
    q_max: float,
    alpha: float = 10.0
) -> float:
    delta = 1e-12
    a = alpha / (cov_90 - cov_10 + delta)
    b = cov_50
    exponent = -a * (cov_value - b)
    sigmoid = 1.0 / (1.0 + np.exp(exponent))
    q_adaptive = q_min + (q_max - q_min) * sigmoid
    return max(q_min, min(q_adaptive, q_max))
