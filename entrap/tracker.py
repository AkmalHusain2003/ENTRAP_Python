import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EntropyProgress:
    cluster_id: int
    candidate_indices: np.ndarray
    mahalanobis_distances: np.ndarray
    entropy_values: np.ndarray
    knee_index: int
    n_accepted: int
    cluster_size_initial: int
    cluster_size_final: int
    accepted_indices: Optional[np.ndarray] = None
    rejected_indices: Optional[np.ndarray] = None


class EntropyProgressTracker:
    def __init__(self):
        self.progress_data: Dict[int, EntropyProgress] = {}
        self._lock = None

    def _ensure_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()

    def record_cluster_progress(
        self,
        cluster_id: int,
        candidate_indices: np.ndarray,
        mahalanobis_distances: np.ndarray,
        entropy_values: np.ndarray,
        knee_index: int,
        cluster_size_initial: int,
        cluster_size_final: int
    ):
        n_accepted = min(knee_index, len(candidate_indices))
        accepted_indices = candidate_indices[:n_accepted].copy()
        rejected_indices = candidate_indices[n_accepted:].copy()

        progress = EntropyProgress(
            cluster_id=cluster_id,
            candidate_indices=np.array(candidate_indices, dtype=np.int64),
            mahalanobis_distances=np.array(mahalanobis_distances, dtype=np.float64),
            entropy_values=np.array(entropy_values, dtype=np.float64),
            knee_index=knee_index,
            n_accepted=n_accepted,
            cluster_size_initial=cluster_size_initial,
            cluster_size_final=cluster_size_final,
            accepted_indices=accepted_indices,
            rejected_indices=rejected_indices
        )

        self._ensure_lock()
        with self._lock:
            self.progress_data[cluster_id] = progress

    def get_cluster_progress(self, cluster_id: int) -> Optional[EntropyProgress]:
        return self.progress_data.get(cluster_id, None)

    def list_clusters(self) -> List[int]:
        return sorted(self.progress_data.keys())

    def get_accepted_indices(self, cluster_id: int) -> Optional[np.ndarray]:
        progress = self.get_cluster_progress(cluster_id)
        if progress is None:
            return None
        return progress.accepted_indices

    def get_rejected_indices(self, cluster_id: int) -> Optional[np.ndarray]:
        progress = self.get_cluster_progress(cluster_id)
        if progress is None:
            return None
        return progress.rejected_indices

    def get_rejected_details(self, cluster_id: int) -> Optional[Dict]:
        progress = self.get_cluster_progress(cluster_id)
        if progress is None or progress.rejected_indices is None:
            return None

        n_accepted = progress.n_accepted
        n_total = len(progress.candidate_indices)
        return {
            'indices': progress.rejected_indices,
            'mahalanobis_distances': progress.mahalanobis_distances[n_accepted:],
            'entropy_values': progress.entropy_values[n_accepted:],
            'count': len(progress.rejected_indices),
            'percentage': 100 * len(progress.rejected_indices) / max(n_total, 1)
        }

    def get_accepted_details(self, cluster_id: int) -> Optional[Dict]:
        progress = self.get_cluster_progress(cluster_id)
        if progress is None or progress.accepted_indices is None:
            return None

        n_accepted = progress.n_accepted
        n_total = len(progress.candidate_indices)
        return {
            'indices': progress.accepted_indices,
            'mahalanobis_distances': progress.mahalanobis_distances[:n_accepted],
            'entropy_values': progress.entropy_values[:n_accepted],
            'count': len(progress.accepted_indices),
            'percentage': 100 * len(progress.accepted_indices) / max(n_total, 1)
        }

    def plot_rejected_analysis(
        self,
        cluster_id: int,
        figsize: Tuple[int, int] = (16, 6),
        save_path: Optional[str] = None
    ):
        progress = self.get_cluster_progress(cluster_id)
        if progress is None:
            raise ValueError(
                f"Cluster {cluster_id} tidak ditemukan. "
                f"Available clusters: {self.list_clusters()}"
            )

        if progress.rejected_indices is None or len(progress.rejected_indices) == 0:
            print(f"Cluster {cluster_id}: Tidak ada kandidat yang ditolak")
            return

        n_accepted = progress.n_accepted
        n_total = len(progress.candidate_indices)
        n_rejected = len(progress.rejected_indices)

        accepted_mahal = progress.mahalanobis_distances[:n_accepted]
        rejected_mahal = progress.mahalanobis_distances[n_accepted:]
        accepted_entropy = progress.entropy_values[:n_accepted]
        rejected_entropy = progress.entropy_values[n_accepted:]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        bins = np.linspace(
            min(progress.mahalanobis_distances.min(), 0),
            progress.mahalanobis_distances.max(),
            30
        )

        ax1 = axes[0]
        ax1.hist(accepted_mahal, bins=bins, alpha=0.6, color='green',
                label=f'Accepted (n={n_accepted})', edgecolor='black')
        ax1.hist(rejected_mahal, bins=bins, alpha=0.6, color='red',
                label=f'Rejected (n={n_rejected})', edgecolor='black')
        if len(accepted_mahal) > 0:
            ax1.axvline(accepted_mahal.max(), color='darkgreen', linestyle='--',
                       linewidth=2, label=f'Max Accepted: {accepted_mahal.max():.3f}')
        if len(rejected_mahal) > 0:
            ax1.axvline(rejected_mahal.min(), color='darkred', linestyle='--',
                       linewidth=2, label=f'Min Rejected: {rejected_mahal.min():.3f}')
        ax1.set_xlabel('Mahalanobis Distance', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(
            f'Cluster {cluster_id}: Mahalanobis Distance\nAccepted vs Rejected',
            fontsize=12, fontweight='bold'
        )
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')

        bins_entropy = np.linspace(
            min(progress.entropy_values.min(), 0),
            progress.entropy_values.max(),
            30
        )

        ax2 = axes[1]
        ax2.hist(accepted_entropy, bins=bins_entropy, alpha=0.6, color='green',
                label=f'Accepted (n={n_accepted})', edgecolor='black')
        ax2.hist(rejected_entropy, bins=bins_entropy, alpha=0.6, color='red',
                label=f'Rejected (n={n_rejected})', edgecolor='black')
        if len(accepted_entropy) > 0:
            ax2.axvline(accepted_entropy.max(), color='darkgreen', linestyle='--',
                       linewidth=2, label=f'Max Accepted: {accepted_entropy.max():.3f}')
        if len(rejected_entropy) > 0:
            ax2.axvline(rejected_entropy.min(), color='darkred', linestyle='--',
                       linewidth=2, label=f'Min Rejected: {rejected_entropy.min():.3f}')
        ax2.set_xlabel('Persistence Entropy', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(
            f'Cluster {cluster_id}: Persistence Entropy\nAccepted vs Rejected',
            fontsize=12, fontweight='bold'
        )
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')

        ax3 = axes[2]
        ax3.scatter(accepted_mahal, accepted_entropy, c='green', s=50, alpha=0.6,
                   label=f'Accepted (n={n_accepted})',
                   edgecolors='black', linewidths=0.5)
        ax3.scatter(rejected_mahal, rejected_entropy, c='red', s=50, alpha=0.6,
                   label=f'Rejected (n={n_rejected})',
                   edgecolors='black', linewidths=0.5)
        if n_accepted > 0 and n_rejected > 0:
            knee_mahal = accepted_mahal.max()
            knee_entropy = accepted_entropy[-1]
            ax3.scatter([knee_mahal], [knee_entropy], c='darkred', s=300,
                       marker='*', edgecolors='black', linewidths=2, zorder=10,
                       label='Knee Point')
        ax3.set_xlabel('Mahalanobis Distance', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Persistence Entropy', fontsize=11, fontweight='bold')
        ax3.set_title(
            f'Cluster {cluster_id}: Mahalanobis vs Entropy\nDecision Boundary',
            fontsize=12, fontweight='bold'
        )
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} - REJECTED CANDIDATES ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Candidates      : {n_total}")
        print(f"Accepted              : {n_accepted} ({100*n_accepted/n_total:.1f}%)")
        print(f"Rejected              : {n_rejected} ({100*n_rejected/n_total:.1f}%)")
        if len(accepted_mahal) > 0:
            print("\nACCEPTED STATS:")
            print(f"  Mahalanobis Range   : [{accepted_mahal.min():.3f}, {accepted_mahal.max():.3f}]")
            print(f"  Mahalanobis Mean    : {accepted_mahal.mean():.3f} \u00b1 {accepted_mahal.std():.3f}")
            print(f"  Entropy Range       : [{accepted_entropy.min():.3f}, {accepted_entropy.max():.3f}]")
            print(f"  Entropy Mean        : {accepted_entropy.mean():.3f} \u00b1 {accepted_entropy.std():.3f}")
        if len(rejected_mahal) > 0:
            print("\nREJECTED STATS:")
            print(f"  Mahalanobis Range   : [{rejected_mahal.min():.3f}, {rejected_mahal.max():.3f}]")
            print(f"  Mahalanobis Mean    : {rejected_mahal.mean():.3f} \u00b1 {rejected_mahal.std():.3f}")
            print(f"  Entropy Range       : [{rejected_entropy.min():.3f}, {rejected_entropy.max():.3f}]")
            print(f"  Entropy Mean        : {rejected_entropy.mean():.3f} \u00b1 {rejected_entropy.std():.3f}")
        print(f"{'='*60}\n")

    def plot_entropy_curve(
        self,
        cluster_id: int,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        progress = self.get_cluster_progress(cluster_id)
        if progress is None:
            raise ValueError(
                f"Cluster {cluster_id} tidak ditemukan. "
                f"Available clusters: {self.list_clusters()}"
            )

        n_candidates = len(progress.entropy_values)
        if n_candidates == 0:
            print(f"Cluster {cluster_id}: Tidak ada kandidat untuk divisualisasikan")
            return

        x_original = np.arange(1, n_candidates + 1)
        entropy_original = progress.entropy_values
        knee_index = progress.knee_index

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_original, entropy_original, 'b-', linewidth=2,
                label='Persistence Entropy', alpha=0.7)
        ax.scatter(x_original, entropy_original, c='blue', s=30, alpha=0.5, zorder=3)

        if knee_index > 0:
            ax.axvspan(0, knee_index, alpha=0.15, color='green', label='Accepted Region')
        if knee_index < n_candidates:
            ax.axvspan(knee_index, n_candidates, alpha=0.15, color='red',
                      label='Rejected Region')

        if 0 < knee_index <= n_candidates:
            knee_entropy = entropy_original[knee_index - 1]
            ax.axvline(x=knee_index, color='darkred', linestyle='--',
                       linewidth=2, alpha=0.8, label='Knee Point')
            ax.scatter([knee_index], [knee_entropy], c='darkred', s=200,
                       marker='*', edgecolors='black', linewidth=1.5, zorder=5)

        ax.set_xlabel(
            'Candidate Order (Nearest \u2192 Farthest by Mahalanobis)',
            fontsize=11, fontweight='bold'
        )
        ax.set_ylabel('Persistence Entropy (H\u2080)', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Cluster {cluster_id}: Sequential Persistence Entropy',
            fontsize=12, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_comparison(
        self,
        cluster_ids: List[int],
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None
    ):
        available_clusters = [cid for cid in cluster_ids if cid in self.progress_data]

        if not available_clusters:
            print(f"Tidak ada cluster valid. Available: {self.list_clusters()}")
            return

        n_clusters = len(available_clusters)
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1) if n_clusters > 1 else [axes]

        for idx, cluster_id in enumerate(available_clusters):
            ax = axes[idx]
            progress = self.progress_data[cluster_id]
            n_candidates = len(progress.entropy_values)
            x_original = np.arange(1, n_candidates + 1)
            entropy_original = progress.entropy_values
            knee_index = progress.knee_index

            ax.plot(x_original, entropy_original, 'b-', linewidth=1.5, alpha=0.7)
            ax.scatter(x_original, entropy_original, c='blue', s=20, alpha=0.4)

            if knee_index > 0:
                ax.axvspan(0, knee_index, alpha=0.15, color='green')

            if 0 < knee_index <= n_candidates:
                knee_entropy = entropy_original[knee_index - 1]
                ax.axvline(x=knee_index, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.8)
                ax.scatter([knee_index], [knee_entropy], c='red', s=100,
                          marker='*', edgecolors='darkred', zorder=5)

            ax.set_xlabel('Candidate Order', fontsize=9)
            ax.set_ylabel('Persistence Entropy', fontsize=9)
            ax.set_title(
                f'Cluster {cluster_id}\nAccepted: {progress.n_accepted}/{n_candidates}',
                fontsize=10, fontweight='bold'
            )
            ax.grid(True, alpha=0.2, linestyle=':')

        for idx in range(len(available_clusters), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Entropy Curves Comparison Across Clusters',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def export_summary(self) -> Dict:
        summary = {}
        for cluster_id, progress in self.progress_data.items():
            summary[cluster_id] = {
                'n_candidates': len(progress.candidate_indices),
                'n_accepted': progress.n_accepted,
                'acceptance_rate': (
                    progress.n_accepted / max(len(progress.candidate_indices), 1)
                ),
                'knee_index': progress.knee_index,
                'cluster_size_initial': progress.cluster_size_initial,
                'cluster_size_final': progress.cluster_size_final,
                'size_change': (
                    progress.cluster_size_final - progress.cluster_size_initial
                ),
                'mean_entropy': float(np.mean(progress.entropy_values)),
                'std_entropy': float(np.std(progress.entropy_values)),
                'entropy_range': (
                    float(np.min(progress.entropy_values)),
                    float(np.max(progress.entropy_values))
                )
            }
        return summary
