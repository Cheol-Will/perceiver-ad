import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.Baselines.Trainer import Trainer

class Analyzer(Trainer):
    def __init__(self, model_config: dict):
        super(Analyzer, self).__init__(model_config)

    @torch.no_grad()
    def plot_anomaly_histograms(
        self,
        bins: int = 50,
        remove_outliers: bool = False,
        outlier_method: str = "percentile",
        low: float = 0.0,
        high: float = 95.0,
        iqr_k: float = 1.5,
    ):
        def _scores_and_labels(dataset):
            X = self._to_numpy(dataset.data)
            scores = self.model.decision_function(X)
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)

            y = getattr(dataset, "targets", None)
            if y is None:
                labels = np.zeros_like(scores, dtype=np.int64)
            else:
                labels = self._to_numpy(y, dtype=np.int64).reshape(-1)
            return scores, labels

        train_scores, train_labels = _scores_and_labels(self.train_set)
        test_scores,  test_labels  = _scores_and_labels(self.test_set)

        train_normal = train_scores[train_labels == 0] if train_labels.size > 0 else train_scores
        test_normal   = test_scores[test_labels == 0] if test_labels.size > 0 else np.array([])
        test_abnormal = test_scores[test_labels != 0] if test_labels.size > 0 else np.array([])

        def _clip(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            arr = arr.astype(np.float64)
            if outlier_method == "percentile":
                lo = np.percentile(arr, low)
                hi = np.percentile(arr, high)
            elif outlier_method == "iqr":
                q1 = np.percentile(arr, 25.0)
                q3 = np.percentile(arr, 75.0)
                iqr = q3 - q1
                lo = q1 - iqr_k * iqr
                hi = q3 + iqr_k * iqr
            else:
                raise ValueError("outlier_method must be 'percentile' or 'iqr'")
            return arr[(arr >= lo) & (arr <= hi)]

        if remove_outliers:
            train_plot = _clip(train_normal)
            test_norm_plot = _clip(test_normal)
            test_abn_plot  = _clip(test_abnormal)
        else:
            train_plot = train_normal
            test_norm_plot = test_normal
            test_abn_plot  = test_abnormal

        all_nonempty = [a for a in [train_plot, test_norm_plot, test_abn_plot] if a.size > 0]
        if len(all_nonempty) == 0:
            raise RuntimeError("No score error: all arrays are empty.")
        global_min = min(a.min() for a in all_nonempty)
        global_max = max(a.max() for a in all_nonempty)
        if global_min == global_max:
            eps = 1e-6 if global_min == 0 else abs(global_min) * 1e-3
            global_min -= eps
            global_max += eps
        bin_edges = np.linspace(global_min, global_max, bins + 1)

        base_path = self.model_config.get('base_path', 'metrics')
        os.makedirs(base_path, exist_ok=True)

        # ----- overlay plot -----
        plt.figure(figsize=(7, 5), dpi=200)
        labels_drawn = []
        if train_plot.size > 0:
            plt.hist(train_plot, bins=bin_edges, alpha=0.45, density=True, label="Train (normal)")
            labels_drawn.append("Train (normal)")
        if test_norm_plot.size > 0:
            plt.hist(test_norm_plot,  bins=bin_edges, alpha=0.45, density=True, label="Test (normal)")
            labels_drawn.append("Test (normal)")
        if test_abn_plot.size > 0:
            plt.hist(test_abn_plot, bins=bin_edges, alpha=0.45, density=True, label="Test (abnormal)")
            labels_drawn.append("Test (abnormal)")
        title_suffix = " (clipped)" if remove_outliers else ""
        ds_name = self.model_config.get('dataset_name', 'DATASET').upper()
        plt.title(f"Anomaly Score on {ds_name}{title_suffix}")
        plt.xlabel("Anomaly score")
        plt.ylabel("Density")
        if labels_drawn:
            plt.legend()
        plt.tight_layout()
        out_overlay = os.path.join(base_path, "hist_anomaly_score.png")
        plt.savefig(out_overlay)
        plt.close()

        # ----- 1x3 grid -----
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200, sharey=True)

        def _plot_single(ax, data: np.ndarray, title: str):
            ax.set_title(title)
            if data.size > 0:
                ax.hist(data, bins=bin_edges, density=True, alpha=0.85)
                ax.set_xlim(global_min, global_max)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Anomaly score")

        _plot_single(axes[0], train_plot, "Train (normal)")
        _plot_single(axes[1], test_norm_plot,  "Test (normal)")
        _plot_single(axes[2], test_abn_plot, "Test (abnormal)")
        axes[0].set_ylabel("Density")

        fig.suptitle(f"Anomaly Score Distributions • {ds_name}{title_suffix}", y=1.02, fontsize=11)
        fig.tight_layout()
        out_grid = os.path.join(base_path, "hist_anomaly_score_1x3.png")
        fig.savefig(out_grid, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved histogram to {out_overlay}, {out_grid}")
        return out_overlay, out_grid
