import numpy as np
from typing import Dict, Tuple
from scipy.spatial import cKDTree

from entrap.constants import DEK_Q_MIN, DEK_Q_MAX, K_MIN, K_MAX, M_MIN, K_PERCENTILE
from entrap.utils import optimize_memory
from entrap.numba_core import compute_cov_from_rows, logistic_mapping
from entrap.intrinsic_dim import estimate_intrinsic_dimension_twenn


def compute_cov_distribution(
    per_point_covs: np.ndarray
) -> Tuple[float, float, float]:
    if len(per_point_covs) == 0:
        return (0.0, 0.5, 1.0)

    finite_mask = np.isfinite(per_point_covs)
    finite_array = per_point_covs[finite_mask]

    if len(finite_array) == 0:
        return (0.0, 0.5, 1.0)

    q10 = float(np.percentile(finite_array, 10))
    q50 = float(np.percentile(finite_array, 50))
    q90 = float(np.percentile(finite_array, 90))

    if abs(q90 - q10) < 1e-12:
        epsilon = max(1e-6, 0.01 * abs(q10) if q10 != 0.0 else 1e-6)
        q90 = q10 + epsilon

    return (q10, q50, q90)


class Density_Equalization_K:
    def __init__(self, alpha: float = 10.0):
        self.q_min = DEK_Q_MIN
        self.q_max = DEK_Q_MAX
        self.k_min = K_MIN
        self.k_max = K_MAX
        self.m_min = M_MIN
        self.alpha = float(alpha)

        self.cluster_k_values_: Dict[int, np.ndarray] = {}
        self.cluster_intrinsic_dims_: Dict[int, float] = {}
        self.cluster_basic_stats_: Dict[int, Dict] = {}
        self.fitted_ = False
        self.last_estimated_dim_ = 1.0

    def _compute_adaptive_m(self, X: np.ndarray, n: int) -> int:
        if n <= 2:
            return max(int(self.m_min), n - 1)

        d_hat = estimate_intrinsic_dimension_twenn(X, X_is_dist=False)
        self.last_estimated_dim_ = d_hat
        m_adaptive = int(np.floor(n ** (1.0 / (d_hat + 1.0))))
        return int(np.clip(m_adaptive, int(self.m_min), n - 1))

    @optimize_memory
    def fit(self, X: np.ndarray, labels: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        labels = labels.astype(int)

        unique_clusters = np.unique(labels[labels >= 0])
        self.cluster_k_values_.clear()
        self.cluster_intrinsic_dims_.clear()
        self.cluster_basic_stats_.clear()

        for cid in unique_clusters:
            mask = (labels == cid)
            cluster_points = X[mask]
            n_points = cluster_points.shape[0]

            if n_points <= 1:
                self.cluster_k_values_[cid] = np.full(n_points, int(self.k_min))
                self.cluster_intrinsic_dims_[cid] = 1.0
                self.cluster_basic_stats_[cid] = {
                    'n_points': n_points,
                    'k_mean': float(self.k_min),
                    'k_std': 0.0
                }
                continue

            m_adaptive = self._compute_adaptive_m(cluster_points, n_points)
            d_hat = self.last_estimated_dim_
            self.cluster_intrinsic_dims_[cid] = d_hat

            k_query = min(m_adaptive + 1, n_points)
            tree = cKDTree(cluster_points, leafsize=40,
                          compact_nodes=True, balanced_tree=True)
            distances, _ = tree.query(cluster_points, k=k_query, p=2, workers=-1)

            neighbor_dist = (
                distances[:, 1:] if k_query > 1
                else np.zeros((n_points, 0), dtype=float)
            )

            if neighbor_dist.shape[1] == 0:
                self.cluster_k_values_[cid] = np.full(n_points, int(self.k_min))
                self.cluster_basic_stats_[cid] = {
                    'n_points': n_points,
                    'k_mean': float(self.k_min),
                    'k_std': 0.0
                }
                continue

            per_point_covs = compute_cov_from_rows(neighbor_dist)
            cov_10, cov_50, cov_90 = compute_cov_distribution(per_point_covs)

            q_adaptive = np.array([
                logistic_mapping(
                    per_point_covs[i], cov_10, cov_50, cov_90,
                    self.q_min, self.q_max, self.alpha
                )
                for i in range(n_points)
            ], dtype=np.float64)

            r_adaptive = np.array([
                float(np.quantile(neighbor_dist[i, :], q_adaptive[i]))
                if neighbor_dist[i, :].size > 0 else 0.0
                for i in range(n_points)
            ], dtype=np.float64)

            r_adaptive = np.maximum(r_adaptive, 1e-12)

            neighbor_lists = tree.query_ball_tree(tree, r=r_adaptive.max(), p=2)
            k_values = np.array([
                sum(
                    1 for j in neighbor_lists[i]
                    if i != j and
                    np.linalg.norm(cluster_points[i] - cluster_points[j]) <= r_adaptive[i]
                )
                for i in range(n_points)
            ], dtype=np.int64)

            k_values = np.clip(k_values, int(self.k_min), int(self.k_max))
            if np.all(k_values == 0):
                k_values[:] = int(self.k_min)

            self.cluster_k_values_[cid] = k_values
            self.cluster_basic_stats_[cid] = {
                'n_points': n_points,
                'k_mean': float(np.mean(k_values)),
                'k_std': float(np.std(k_values)),
            }

        self.fitted_ = True
        return self

    def get_k_percentile(self, cluster_id: int, percentile: float = K_PERCENTILE) -> int:
        if not self.fitted_ or cluster_id not in self.cluster_k_values_:
            return int(self.k_min)
        vals = self.cluster_k_values_[cluster_id]
        return (
            int(np.round(np.percentile(vals, percentile)))
            if vals.size > 0 else int(self.k_min)
        )

    def get_intrinsic_dimension(self, cluster_id: int) -> float:
        return (
            self.cluster_intrinsic_dims_.get(cluster_id, 1.0)
            if self.fitted_ else 1.0
        )
