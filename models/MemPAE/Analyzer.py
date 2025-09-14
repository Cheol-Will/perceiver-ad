import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from DataSet.DataLoader import get_dataloader
from models.MemPAE.Model import MemPAE
from models.MemPAE.Trainer import Trainer
from utils import aucPerformance, F1Performance
import math

class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict, analysis_config: dict):
        super().__init__(model_config, train_config)

        self.plot_attn = analysis_config['plot_attn'] 
        self.plot_recon = analysis_config['plot_recon'] 
        self.cum_memory_weight = True
        self.model_config = model_config
        self.train_config = train_config
        self.analysis_config = analysis_config
    
    @torch.no_grad()
    def plot_latent_diff(
        self,
    ):
        _, memory_weights_dict = self._accumulate_latent_diff()
        
        train_normal_latent_losses = memory_weights_dict['train_normal_latent_losses']
        test_normal_latent_losses = memory_weights_dict['test_normal_latent_losses']
        test_abnormal_latent_losses = memory_weights_dict['test_abnormal_latent_losses']

        base_path = self.train_config['base_path']
        save_dir = os.path.join(base_path, 'memory_usage_analysis') 
        os.makedirs(save_dir, exist_ok=True)

        return

    @torch.no_grad()
    def _accumulate_latent_diff(
        self,
    ):
        self.model.eval()

        def _collect_latent_diff(loader):
            scores, latent_losses, labels = [], [], []
            for x, y in loader:
                x = x.to(self.device)
                loss, z, z_hat = self.model(x, return_latents=True)
                latent_loss = F.mse_loss(z_hat, z, reduction='none').mean(dim=[1,2]) # (B, )

                
                # loss
                if isinstance(loss, torch.Tensor):
                    loss = loss.view(-1).detach().cpu().float().numpy()
                else:
                    loss = np.asarray(loss, dtype=np.float32).reshape(-1)
                
                # memory weight
                if isinstance(latent_loss, torch.Tensor):
                    latent_loss = latent_loss.view(-1).detach().cpu().float().numpy() # (B, N, M)
                else:
                    latent_loss = np.asarray(latent_loss, dtype=np.float32).reshape(-1)
                
                # label
                if y is None:
                    y_arr = np.zeros(loss.shape[0], dtype=np.int64)
                else:
                    y_arr = y.view(-1).detach().cpu().numpy()
                
                scores.append(loss)
                latent_losses.append(latent_loss)
                labels.append(y_arr)

            if len(scores) == 0:
                return np.array([]), np.array([])
            return np.concatenate(scores, axis=0), np.concatenate(latent_losses, axis=0), np.concatenate(labels, axis=0)

        train_scores, latent_losses, train_labels = _collect_latent_diff(self.train_loader)
        test_scores,  latent_losses, test_labels  = _collect_latent_diff(self.test_loader)
        print(train_scores.shape, latent_losses.shape, train_labels.shape)
        print(test_scores.shape,  latent_losses.shape, test_labels.shape)


        train_normal_scores = train_scores[train_labels == 0] if train_labels.size > 0 else train_scores
        test_normal_scores   = test_scores[test_labels == 0] if test_labels.size > 0 else np.array([])
        test_abnormal_scores = test_scores[test_labels != 0] if test_labels.size > 0 else np.array([])

        train_normal_latent_losses = latent_losses[train_labels == 0] if train_labels.size > 0 else train_scores
        test_normal_latent_losses   = latent_losses[test_labels == 0] if test_labels.size > 0 else np.array([])
        test_abnormal_latent_losses = latent_losses[test_labels != 0] if test_labels.size > 0 else np.array([])

        scores = {
            'train_normal_scores': train_normal_scores,
            'test_normal_scores': test_normal_scores,
            'test_abnormal_scores': test_abnormal_scores,
        }

        memory_weights = {
            'train_normal_latent_losses': train_normal_latent_losses,
            'test_normal_latent_losses': test_normal_latent_losses,
            'test_abnormal_latent_losses': test_abnormal_latent_losses,
        }
        return scores, memory_weights

        

    @torch.no_grad()
    def plot_memory_weight(
        self,
    ):
        _, memory_weights_dict = self._accumulate_memory_weight()
        
        train_normal_weights = memory_weights_dict['train_normal_memory_weights']
        test_normal_weights = memory_weights_dict['test_normal_memory_weights']
        test_abnormal_weights = memory_weights_dict['test_abnormal_memory_weights']

        if train_normal_weights.ndim < 3:
            print("Warning: Not enough data to plot memory weight analysis.")
            return

        num_samples, num_latents, num_memories = train_normal_weights.shape
        
        base_path = self.train_config['base_path']
        save_dir = os.path.join(base_path, 'memory_usage_analysis') 
        os.makedirs(save_dir, exist_ok=True)

        for i in range(num_latents):
            
            train_latent_usage = train_normal_weights[:, i, :].mean(axis=0)
            test_norm_latent_usage = test_normal_weights[:, i, :].mean(axis=0) if test_normal_weights.size > 0 else np.array([])
            test_abn_latent_usage = test_abnormal_weights[:, i, :].mean(axis=0) if test_abnormal_weights.size > 0 else np.array([])

            fig_bar, axes_bar = plt.subplots(1, 3, figsize=(15, 4), dpi=200, sharey=True)

            def _plot_usage_bar(ax, data: np.ndarray, title: str):
                ax.set_title(title)
                if data.size > 0:
                    mem_indices = np.arange(data.shape[0])
                    ax.bar(mem_indices, data, width=1.0, alpha=0.85)
                    if data.shape[0] > 30:
                        ticks = np.linspace(0, data.shape[0] - 1, 5, dtype=int)
                        ax.set_xticks(ticks)
                    ax.set_xlim(-0.5, data.shape[0]-0.5)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("Memory Index")

            _plot_usage_bar(axes_bar[0], train_latent_usage, "Train (normal)")
            _plot_usage_bar(axes_bar[1], test_norm_latent_usage, "Test (normal)")
            _plot_usage_bar(axes_bar[2], test_abn_latent_usage, "Test (abnormal)")
            axes_bar[0].set_ylabel("Mean Weight (Frequency)")

            fig_bar.suptitle(f"Mean Memory Slot Usage for Latent {i} • {self.train_config['dataset_name'].upper()}", y=1.02, fontsize=11)
            fig_bar.tight_layout()
            bar_path = os.path.join(save_dir, f"latent_{i}_memory_usage.png")
            fig_bar.savefig(bar_path, bbox_inches="tight")
            plt.close(fig_bar)
            print(f"Saved Latent {i} memory usage bar plot to {bar_path}")

        train_total_usage = train_normal_weights.mean(axis=(0, 1))
        test_norm_total_usage = test_normal_weights.mean(axis=(0, 1)) if test_normal_weights.size > 0 else np.array([])
        test_abn_total_usage = test_abnormal_weights.mean(axis=(0, 1)) if test_abnormal_weights.size > 0 else np.array([])

        fig_total_bar, axes_total_bar = plt.subplots(1, 3, figsize=(15, 4), dpi=200, sharey=True)

        _plot_usage_bar(axes_total_bar[0], train_total_usage, "Train (normal)")
        _plot_usage_bar(axes_total_bar[1], test_norm_total_usage, "Test (normal)")
        _plot_usage_bar(axes_total_bar[2], test_abn_total_usage, "Test (abnormal)")
        axes_total_bar[0].set_ylabel("Mean Weight (Frequency)")

        fig_total_bar.suptitle(f"Total Mean Memory Slot Usage (All Latents) • {self.train_config['dataset_name'].upper()}", y=1.02, fontsize=11)
        fig_total_bar.tight_layout()
        total_bar_path = os.path.join(save_dir, "total_memory_usage.png")
        fig_total_bar.savefig(total_bar_path, bbox_inches="tight")
        plt.close(fig_total_bar)
        print(f"Saved total memory usage bar plot to {total_bar_path}")

        return

    @torch.no_grad()
    def _accumulate_memory_weight(
        self,
    ):
        self.model.eval()

        def _collect_memory_weight(loader):
            scores, memory_weights, labels = [], [], []
            for x, y in loader:
                x = x.to(self.device)
                loss, memory_weight = self.model(x, return_memory_weight=True)
                
                # loss
                if isinstance(loss, torch.Tensor):
                    loss = loss.view(-1).detach().cpu().float().numpy()
                else:
                    loss = np.asarray(loss, dtype=np.float32).reshape(-1)
                
                # memory weight
                if isinstance(memory_weight, torch.Tensor):
                    memory_weight = memory_weight.detach().cpu().float().numpy() # (B, N, M)
                else:
                    memory_weight = np.asarray(memory_weight, dtype=np.float32).reshape(-1)
                
                # label
                if y is None:
                    y_arr = np.zeros(loss.shape[0], dtype=np.int64)
                else:
                    y_arr = y.view(-1).detach().cpu().numpy()
                
                scores.append(loss)
                memory_weights.append(memory_weight)
                labels.append(y_arr)

            if len(scores) == 0:
                return np.array([]), np.array([])
            return np.concatenate(scores, axis=0), np.concatenate(memory_weights, axis=0), np.concatenate(labels, axis=0)

        train_scores, train_memory_weights, train_labels = _collect_memory_weight(self.train_loader)
        test_scores,  test_memory_weights, test_labels  = _collect_memory_weight(self.test_loader)
        print(train_scores.shape, train_memory_weights.shape, train_labels.shape)
        print(test_scores.shape,  test_memory_weights.shape, test_labels.shape)


        train_normal_scores = train_scores[train_labels == 0] if train_labels.size > 0 else train_scores
        test_normal_scores   = test_scores[test_labels == 0] if test_labels.size > 0 else np.array([])
        test_abnormal_scores = test_scores[test_labels != 0] if test_labels.size > 0 else np.array([])

        train_normal_memory_weights = train_memory_weights[train_labels == 0] if train_labels.size > 0 else train_scores
        test_normal_memory_weights   = test_memory_weights[test_labels == 0] if test_labels.size > 0 else np.array([])
        test_abnormal_memory_weights = test_memory_weights[test_labels != 0] if test_labels.size > 0 else np.array([])

        scores = {
            'train_normal_scores': train_normal_scores,
            'test_normal_scores': test_normal_scores,
            'test_abnormal_scores': test_abnormal_scores,
        }

        memory_weights = {
            'train_normal_memory_weights': train_normal_memory_weights,
            'test_normal_memory_weights': test_normal_memory_weights,
            'test_abnormal_memory_weights': test_abnormal_memory_weights,
        }
        return scores, memory_weights

    @torch.no_grad()
    def _accumulate_attn(
        self,
        attns,                 # List[Tensor], self-attn blocks +  cross-attn
        input_col_idx,         # List[int]
        target_col_idx,        # List[int]
        self_attention_map,    # (F,F) Tensor on device
        self_count_map,        # (F,F) Tensor on device
        cross_attention_map,   # (F,F) Tensor on device
        cross_count_map,       # (F,F) Tensor on device
    ):
        device = self.device
        in_idx  = torch.as_tensor(input_col_idx,  device=device, dtype=torch.long)
        tgt_idx = torch.as_tensor(target_col_idx, device=device, dtype=torch.long)

        if len(attns) > 1:
            self_blocks = attns[:-1]
            self_mean = torch.zeros((in_idx.numel(), in_idx.numel()), device=device)
            for a in self_blocks:
                self_mean = self_mean + a.mean(dim=(0, 1))  # (S,S)
            self_mean = self_mean / len(self_blocks)

            idx_row = in_idx.unsqueeze(1)  # (S,1)
            idx_col = in_idx.unsqueeze(0)  # (1,S)
            self_attention_map.index_put_((idx_row, idx_col), self_mean, accumulate=True)
            self_count_map.index_put_((idx_row, idx_col), torch.ones_like(self_mean), accumulate=True)

        cross = attns[-1]                # (B,H,T,S)
        cross_mean = cross.mean(dim=(0, 1))  # (T,S)

        idx_row = tgt_idx.unsqueeze(1)   # (T,1)
        idx_col = in_idx.unsqueeze(0)    # (1,S)
        cross_attention_map.index_put_((idx_row, idx_col), cross_mean, accumulate=True)
        cross_count_map.index_put_((idx_row, idx_col), torch.ones_like(cross_mean), accumulate=True)

    @torch.no_grad()
    def _finalize_and_save_attention_maps(
        self,
        self_attention_map: torch.Tensor,   # (F,F)
        self_count_map: torch.Tensor,       # (F,F)
        cross_attention_map: torch.Tensor,  # (F,F)
        cross_count_map: torch.Tensor,      # (F,F)
    ):
        self_avg = torch.where(self_count_map > 0, self_attention_map / self_count_map, torch.zeros_like(self_attention_map))
        cross_avg = torch.where(cross_count_map > 0, cross_attention_map / cross_count_map, torch.zeros_like(cross_attention_map))

        self_avg_np  = self_avg.detach().float().cpu().numpy()
        cross_avg_np = cross_avg.detach().float().cpu().numpy()

        base_path = self.model_config['base_path']
        os.makedirs(base_path, exist_ok=True)

        def _save_heatmap(arr: np.ndarray, title: str, out_path: str):
            plt.figure(figsize=(6, 5), dpi=200)
            im = plt.imshow(arr, aspect='auto')
            plt.title(title)
            plt.xlabel('Input feature index')
            plt.ylabel('Output/Query feature index')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

        _save_heatmap(self_avg_np,  'Self-Attention (avg)',  os.path.join(base_path, 'self_attention_map.png'))
        _save_heatmap(cross_avg_np, 'Cross-Attention (avg)', os.path.join(base_path, 'cross_attention_map.png'))

    @torch.no_grad()
    def plot_reconstruction(self):
        self.model.eval()

        def _gather_class_samples(loader, need_per_class=4):
            xs_normal, ys_normal = [], []
            xs_abnorm, ys_abnorm = [], []
            for xb, yb in loader:
                if yb.ndim > 1:
                    yb = yb.squeeze(-1)
                normal_mask = (yb == 0)
                abnorm_mask = (yb != 0)
                if normal_mask.any():
                    xs_normal.append(xb[normal_mask])
                    ys_normal.append(yb[normal_mask])
                if abnorm_mask.any():
                    xs_abnorm.append(xb[abnorm_mask])
                    ys_abnorm.append(yb[abnorm_mask])
                if sum(x.shape[0] for x in xs_normal) >= need_per_class and \
                sum(x.shape[0] for x in xs_abnorm) >= need_per_class:
                    break
            x_norm = torch.cat(xs_normal, dim=0) if xs_normal else torch.empty(0)
            y_norm = torch.cat(ys_normal, dim=0) if ys_normal else torch.empty(0, dtype=torch.long)
            x_abn  = torch.cat(xs_abnorm,  dim=0) if xs_abnorm else torch.empty(0)
            y_abn  = torch.cat(ys_abnorm,  dim=0) if ys_abnorm else torch.empty(0, dtype=torch.long)
            return x_norm, y_norm, x_abn, y_abn

        # gather 4 normal + 4 abnormal
        need = 4
        x_norm, y_norm, x_abn, y_abn = _gather_class_samples(self.test_loader, need_per_class=need)
        if x_norm.shape[0] < need or x_abn.shape[0] < need:
            x_norm2, y_norm2, x_abn2, y_abn2 = _gather_class_samples(self.train_loader, need_per_class=need)
            if x_norm.shape[0] < need and x_norm2.numel() > 0:
                x_norm = torch.cat([x_norm, x_norm2], dim=0)
                y_norm = torch.cat([y_norm, y_norm2], dim=0)
            if x_abn.shape[0] < need and x_abn2.numel() > 0:
                x_abn = torch.cat([x_abn, x_abn2], dim=0)
                y_abn = torch.cat([y_abn, y_abn2], dim=0)

        if x_norm.shape[0] == 0 and x_abn.shape[0] == 0:
            raise RuntimeError("Not found: Normal/Abnormal.")

        def _pad_with_repeat(x, y, target_k):
            if x.shape[0] >= target_k:
                return x[:target_k], y[:target_k]
            idx = torch.arange(x.shape[0])
            rep = (target_k + x.shape[0] - 1) // x.shape[0]
            idx_full = idx.repeat(rep)[:target_k]
            return x[idx_full], y[idx_full]

        if x_norm.shape[0] < need:
            x_norm, y_norm = _pad_with_repeat(x_norm, y_norm, need)
        else:
            x_norm, y_norm = x_norm[:need], y_norm[:need]

        if x_abn.shape[0] < need:
            x_abn, y_abn = _pad_with_repeat(x_abn, y_abn, need)
        else:
            x_abn, y_abn = x_abn[:need], y_abn[:need]

        x = torch.cat([x_norm, x_abn], dim=0).to(self.device)  # (8, F)
        y = torch.cat([y_norm, y_abn], dim=0).to(self.device)  # (8,)

        batch_size, num_features = x.shape
        H = int(np.sqrt(num_features))
        assert H * H == num_features, f"{num_features} is not squared number."


        _, x, recon = self.model(x, return_pred=True)

        base_path = self.model_config['base_path']
        os.makedirs(base_path, exist_ok=True)

        def _norm_with_ref(arr, ref_min, ref_max):
            if ref_max > ref_min:
                return (arr - ref_min) / (ref_max - ref_min)
            return np.zeros_like(arr)

        saved_paths = []
        for i in range(batch_size):
            orig_img  = x[i].detach().float().cpu().numpy().reshape(H, H)
            recon_img = recon[i].detach().float().cpu().numpy().reshape(H, H)

            ref_min, ref_max = orig_img.min(), orig_img.max()
            orig_plot  = _norm_with_ref(orig_img,  ref_min, ref_max)
            recon_plot = _norm_with_ref(recon_img, ref_min, ref_max)

            plt.figure(figsize=(9, 3), dpi=200)
            plt.subplot(1, 2, 1)
            plt.imshow(orig_plot, cmap='gray', vmin=0, vmax=1)
            plt.title(f'Original (label={int(y[i].detach().cpu())})')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(recon_plot, cmap='gray', vmin=0, vmax=1)
            plt.title('Reconstruction')
            plt.axis('off')

            plt.tight_layout()
            out_path = os.path.join(base_path, f'reconstruction_pair_{i}_label{int(y[i].detach().cpu())}.png')
            plt.savefig(out_path)
            plt.close()

            saved_paths.append(out_path)
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.info(f"[plot_reconstruction] saved: {out_path}")

        return {
            "num_saved": len(saved_paths),
            "paths": saved_paths,
            "labels": [int(v) for v in y.detach().cpu().tolist()],
        }
   
    @torch.no_grad()
    def plot_anomaly_histograms(
        self,
        bins: int = 50,
        remove_outliers: bool = False,
        outlier_method: str = "percentile",
        low: float = 0.0,
        high: float = 90.0,
        iqr_k: float = 1.5,
    ):
        self.model.eval()

        def _collect_scores(loader):
            scores, labels = [], []
            for xb, yb in loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                if isinstance(out, torch.Tensor):
                    out = out.view(-1).detach().cpu().float().numpy()
                else:
                    out = np.asarray(out, dtype=np.float32).reshape(-1)
                scores.append(out)

                if yb is None:
                    y_arr = np.zeros(out.shape[0], dtype=np.int64)
                else:
                    y_arr = yb.view(-1).detach().cpu().numpy()
                labels.append(y_arr)
            if len(scores) == 0:
                return np.array([]), np.array([])
            return np.concatenate(scores, axis=0), np.concatenate(labels, axis=0)

        train_scores, train_labels = _collect_scores(self.train_loader)
        test_scores,  test_labels  = _collect_scores(self.test_loader)

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
            raise RuntimeError("No score error.")
        global_min = min(a.min() for a in all_nonempty)
        global_max = max(a.max() for a in all_nonempty)
        if global_min == global_max:
            eps = 1e-6 if global_min == 0 else abs(global_min) * 1e-3
            global_min -= eps
            global_max += eps
        bin_edges = np.linspace(global_min, global_max, bins + 1)

        base_path = self.train_config['base_path']
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
        plt.title(f"Anomaly Score on {self.train_config['dataset_name'].upper()}{title_suffix}")
        plt.xlabel("Anomaly score")
        plt.ylabel("Density")
        if labels_drawn:
            plt.legend()
        plt.tight_layout()
        out_overlay = os.path.join(base_path, "hist_anomaly_score.png")
        plt.savefig(out_overlay)
        plt.close()

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

        fig.suptitle(f"Anomaly Score Distributions • {self.train_config['dataset_name'].upper()}{title_suffix}",
                    y=1.02, fontsize=11)
        fig.tight_layout()
        out_grid = os.path.join(base_path, "hist_anomaly_score_1x3.png")
        fig.savefig(out_grid, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved histogram to {out_overlay}, {out_grid}")