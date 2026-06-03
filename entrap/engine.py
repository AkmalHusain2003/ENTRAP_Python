import gc
import numpy as np
import multiprocessing
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

from entrap.constants import RIDGE_EPSILON, K_MIN
from entrap.utils import validate_metric, optimize_memory
from entrap.numba_core import (
    compute_cluster_mean,
    compute_cluster_covariance,
    compute_mahalanobis_sq
)
from entrap.tda import (
    compute_h0_diagram,
    compute_persistence_entropy,
    compute_sequential_persistence_entropy,
    detect_knee_with_kneed
)
from entrap.tracker import EntropyProgressTracker
from entrap.dek import Density_Equalization_K

def _evaluate_cluster_worker(
    X: np.ndarray,
    labels: np.ndarray,
    cid: int,
    candidates_set: set,
    ridge_epsilon: float,
    metric: str
) -> Tuple[Dict, Optional[Dict]]:
    
    if len(candidates_set) == 0:
        return {}, None

    # Filter: hanya kandidat yang masih berstatus noise (label == -1)
    candidates_list = [idx for idx in candidates_set if labels[idx] == -1]
    if len(candidates_list) == 0:
        return {}, None

    # Ekstrak cluster points
    cluster_mask = labels == cid
    cluster_points = X[cluster_mask]
    cluster_size_initial = len(cluster_points)
    d = cluster_points.shape[1]

    # Hitung initial entropy sebelum kandidat ditambahkan
    entropy_initial = compute_persistence_entropy(
        compute_h0_diagram(cluster_points, metric=metric)
    )

    # Hitung cluster statistics untuk Mahalanobis distance
    mu = compute_cluster_mean(cluster_points)
    Sigma_reg = compute_cluster_covariance(cluster_points, mu, ridge_epsilon)

    try:
        Sigma_inv = np.linalg.inv(Sigma_reg)
        log_det_Sigma = np.linalg.slogdet(Sigma_reg)[1]
    except np.linalg.LinAlgError:
        # Fallback: spherical covariance (kondisi degenerate)
        Sigma_inv = np.eye(d) / (np.trace(Sigma_reg) / d + 1e-6)
        log_det_Sigma = 0.0

    candidate_distances = []
    for idx in candidates_list:
        diff = X[idx] - mu
        mahal_sq = compute_mahalanobis_sq(diff, Sigma_inv)
        dist = (
            0.5 * mahal_sq
            + 0.5 * log_det_Sigma
            + (d / 2.0) * np.log(2 * np.pi)
        )
        candidate_distances.append((idx, float(dist)))

    candidate_distances.sort(key=lambda item: item[1])

    sorted_indices = np.array([idx for idx, _ in candidate_distances])
    sorted_mahalanobis = np.array([dist for _, dist in candidate_distances])
    sorted_candidates = X[sorted_indices]

    entropy_values, _ = compute_sequential_persistence_entropy(
        cluster_points,
        sorted_candidates,
        sorted_indices,
        metric=metric
    )

    cutoff_index = detect_knee_with_kneed(entropy_values)

    cluster_eval = {}
    for i, (idx, dist) in enumerate(candidate_distances):
        accepted = (i < cutoff_index)
        entropy_after = entropy_values[i] if i < len(entropy_values) else np.inf
        entropy_before = entropy_initial if i == 0 else entropy_values[i - 1]
        delta_entropy = entropy_after - entropy_before

        cluster_eval[idx] = {
            'mahalanobis_distance': dist,
            'persistence_entropy': entropy_after,
            'entropy_before': entropy_before,
            'delta_entropy': delta_entropy,
            'accepted': accepted,
            'order': i
        }

    tracking_data = {
        'cluster_id': cid,
        'candidate_indices': sorted_indices,
        'mahalanobis_distances': sorted_mahalanobis,
        'entropy_values': entropy_values,
        'knee_index': cutoff_index,
        'cluster_size_initial': cluster_size_initial,
        'cluster_size_final': cluster_size_initial
    }

    gc.collect()
    return cluster_eval, tracking_data


class Geometric_Persistence_Entropy_Engine:
    def __init__(
        self,
        ridge_epsilon: float = RIDGE_EPSILON,
        metric: str = 'euclidean',
        use_memmap: bool = True,
        enable_tracking: bool = False,
        **metric_params
    ):
        self.ridge_epsilon = ridge_epsilon
        self.metric = validate_metric(metric)
        self.metric_params = metric_params
        self.use_memmap = use_memmap
        self.enable_tracking = enable_tracking
        self.tracker = EntropyProgressTracker() if enable_tracking else None

    def _identify_candidates(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        cluster_ids: List[int],
        dek_selector: Optional[Density_Equalization_K]
    ) -> Dict[int, set]:

        cluster_candidate_sets = {}
        noise_indices = np.where(labels == -1)[0]

        if len(noise_indices) == 0:
            return {cid: set() for cid in cluster_ids}

        noise_points = X[noise_indices]
        noise_tree = cKDTree(noise_points, compact_nodes=True, balanced_tree=True)

        for cid in cluster_ids:
            cluster_mask = labels == cid
            cluster_points = X[cluster_mask]

            k_adaptive = (
                dek_selector.get_k_percentile(cid)
                if dek_selector is not None else int(K_MIN)
            )
            k_query = min(k_adaptive, len(cluster_points))

            distances, indices = noise_tree.query(cluster_points, k=k_query, workers=-1)

            if cluster_points.shape[0] == 1:
                distances = distances.reshape(1, -1)
                indices = indices.reshape(1, -1)

            candidate_local_indices = np.unique(indices.ravel())
            candidate_local_indices = candidate_local_indices[
                candidate_local_indices < len(noise_points)
            ]

            if len(candidate_local_indices) > 0:
                candidate_global_indices = noise_indices[candidate_local_indices]
                cluster_candidate_sets[cid] = set(
                    int(idx) for idx in candidate_global_indices
                )
            else:
                cluster_candidate_sets[cid] = set()

        return cluster_candidate_sets

    def _resolve_conflicts(
        self,
        labels: np.ndarray,
        cluster_evaluations: Dict[int, Dict]
    ) -> Tuple[np.ndarray, int]:

        refined_labels = labels.copy()
        total_rescued = 0

        all_accepted = set()
        for cid, evals in cluster_evaluations.items():
            for idx, info in evals.items():
                if info['accepted']:
                    all_accepted.add(idx)

        for candidate_idx in all_accepted:
            if refined_labels[candidate_idx] != -1:
                continue

            competing_clusters = []
            for cid, evals in cluster_evaluations.items():
                if candidate_idx in evals and evals[candidate_idx]['accepted']:
                    delta_entropy = evals[candidate_idx]['delta_entropy']
                    competing_clusters.append((cid, delta_entropy))

            if len(competing_clusters) == 0:
                continue

            winner_cid = min(competing_clusters, key=lambda x: x[1])[0]
            refined_labels[candidate_idx] = winner_cid
            total_rescued += 1

        return refined_labels, total_rescued

    def _compute_final_stats(
        self,
        labels: np.ndarray,
        cluster_ids: List[int],
        cluster_evaluations: Dict[int, Dict]
    ) -> Dict:
        cluster_stats = {}
        for cid in cluster_ids:
            evals = cluster_evaluations.get(cid, {})
            rescued = sum(
                1 for idx, info in evals.items()
                if info['accepted'] and labels[idx] == cid
            )
            cluster_stats[int(cid)] = {
                'rescued': rescued,
                'candidates_evaluated': len(evals),
                'final_size': int(np.sum(labels == cid))
            }
        return cluster_stats

    def reassign_parallel(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        dek_selector: Optional[Density_Equalization_K] = None,
        n_jobs: int = -1
    ) -> Tuple[np.ndarray, int, Dict]:

        refined_labels = labels.copy()
        noise_mask = refined_labels == -1

        if not noise_mask.any():
            return refined_labels, 0, {}

        unique_labels = np.unique(refined_labels[refined_labels >= 0])
        if len(unique_labels) == 0:
            return refined_labels, 0, {}

        # Sort clusters by size descending
        cluster_sizes = [
            (int(cid), int(np.sum(refined_labels == cid)))
            for cid in unique_labels
        ]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_cluster_ids = [cid for cid, _ in cluster_sizes]

        cluster_candidate_sets = self._identify_candidates(
            X, refined_labels, sorted_cluster_ids, dek_selector
        )

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        results_list = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_evaluate_cluster_worker)(
                X,
                refined_labels,
                cid,
                cluster_candidate_sets[cid],
                self.ridge_epsilon,
                self.metric
            )
            for cid in sorted_cluster_ids
        )

        cluster_evaluations = {}
        tracking_data_list = []
        for cid, (cluster_eval, tracking_data) in zip(sorted_cluster_ids, results_list):
            cluster_evaluations[cid] = cluster_eval
            if tracking_data is not None:
                tracking_data_list.append((cid, tracking_data))

        refined_labels, total_rescued = self._resolve_conflicts(
            refined_labels, cluster_evaluations
        )

        if self.enable_tracking and self.tracker is not None:
            for cid, td in tracking_data_list:
                td['cluster_size_final'] = int(np.sum(refined_labels == cid))
                self.tracker.record_cluster_progress(**td)

        cluster_stats = self._compute_final_stats(
            refined_labels, sorted_cluster_ids, cluster_evaluations
        )

        return refined_labels, total_rescued, cluster_stats

    @optimize_memory
    def reassign(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        dek_selector: Optional[Density_Equalization_K] = None
    ) -> Tuple[np.ndarray, int, Dict]:
        return self.reassign_parallel(X, labels, dek_selector, n_jobs=1)