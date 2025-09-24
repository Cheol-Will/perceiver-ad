import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from models.MemPAE.Trainer import Trainer



class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict, analysis_config: dict):
        super().__init__(model_config, train_config)

        # self.plot_attn = analysis_config['plot_attn'] 
        self.plot_recon = analysis_config['plot_recon'] 
        self.cum_memory_weight = True
        self.model_config = model_config
        self.train_config = train_config
        self.analysis_config = analysis_config

    def training(self):
        print(self.model_config)
        print(self.train_config)
        parameter_path = os.path.join(self.train_config['base_path'], 'model.pt')
        if os.path.exists(parameter_path):
            print(f"model.pt already exists at {parameter_path}. Skip training and load parameters.")
            
            self.model.load_state_dict(torch.load(parameter_path))  # 
            self.model.eval()
            return

        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")

        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                loss = self.model(x_input).mean() # (B) -> scalar

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            self.logger.info(info.format(epoch,loss.cpu()))
        print("Training complete.")

        print("Saving")
        torch.save(self.model.state_dict(), parameter_path)

    def plot_pos_encoding(self, use_mask = False):
        if use_mask: 
            pos_encoding = self.model.decoder_query + self.model.pos_encoding  # (1, F, D)
        else:
            pos_encoding = self.model.pos_encoding  # (1, F, D)
        
        if pos_encoding.dim() == 3:
            pos_encoding = pos_encoding.squeeze(0)  # (F, D)
        
        if isinstance(pos_encoding, torch.Tensor):
            pos_encoding_np = pos_encoding.detach().cpu().numpy()
        else:
            pos_encoding_np = pos_encoding
        
        pos_encoding_normalized = pos_encoding_np / (np.linalg.norm(pos_encoding_np, axis=1, keepdims=True) + 1e-8)
        cosine_sim_matrix = np.dot(pos_encoding_normalized, pos_encoding_normalized.T)
        
        plt.figure(figsize=(10, 8), dpi=200)
        im = plt.imshow(cosine_sim_matrix, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
        
        plt.title('Position Encoding Cosine Similarity Matrix', fontsize=14, pad=20)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Feature Index', fontsize=12)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
        plt.tight_layout()
        
        base_path = self.train_config['base_path']
        file_name = f"pos_encoding+token_{self.train_config['dataset_name']}.png" if use_mask else f"pos_encoding_{self.train_config['dataset_name']}.png"
        out_path = os.path.join(base_path, file_name)

        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"pos_encoding saved into '{out_path}'.")
        
        return


    @torch.no_grad()
    def compare_regresssion_with_attn(self):
        attn_with_self, label = self.get_attn_weights(use_self_attn=True)
        attn_no_self, label = self.get_attn_weights(use_self_attn=False)

        attn_label_with_self = attn_with_self[:, -1, :-1].cpu().numpy()
        attn_label_no_self = attn_no_self[:, -1, :-1].cpu().numpy()

        X_all, y_all = [], []
        for X, y in self.train_loader:
            X_all.append(X[:, :-1])
            y_all.append(X[:, -1])
        X_all = torch.cat(X_all, dim=0).cpu().numpy()
        y_all = torch.cat(y_all, dim=0).cpu().numpy()
        
        reg = LinearRegression().fit(X_all, y_all)
        coef = reg.coef_

        num_heads = attn_with_self.shape[0]
        head_cols_self = [f'head{i+1}_self' for i in range(num_heads)]
        head_cols_no_self = [f'head{i+1}_no_self' for i in range(num_heads)]
        columns = ['reg'] + head_cols_self + head_cols_no_self

        data = np.vstack([coef, attn_label_with_self, attn_label_no_self])
        df = pd.DataFrame(data.T, columns=columns)
        num_features = df.shape[0]
        df.index = [f'feature_{i}' for i in range(num_features)]

        df['reg_abs'] = df['reg'].abs()
        df['reg_abs'] = df['reg_abs'] / df['reg_abs'].sum()
        df['head_sum_self'] = df[head_cols_self].abs().sum(axis=1)
        df['head_sum_no_self'] = df[head_cols_no_self].abs().sum(axis=1)

        cols_order = ['reg', 'reg_abs'] + head_cols_self + ['head_sum_self'] + head_cols_no_self + ['head_sum_no_self']
        df = df[cols_order]

        reg_abs_values = df['reg_abs'].values
        cos_sim_list, rank_corr_list = [], []
        
        for col in df.columns:
            col_values = df[col].values
            sim = cosine_similarity(reg_abs_values.reshape(1, -1), col_values.reshape(1, -1))[0, 0]
            cos_sim_list.append(sim)
            rho, _ = spearmanr(reg_abs_values, col_values)
            rank_corr_list.append(rho)

        df.loc['rank_correlation', :] = rank_corr_list
        df.loc['cosine_similarity', :] = cos_sim_list
        df = df.round(4)

        base_path = self.train_config['base_path']
        path = os.path.join(base_path, 'attn_regression_comparison.csv')
        df.to_csv(path, index=True)
        
        print('Attention vs. Regression Comparison\n', df)
        return df

    @torch.no_grad()
    def get_attn_weights(
        self,
        use_self_attn: bool = True,
    ):
        """
        enc_attn: mean over heads
        self_attn: keep per head (compose residualized layers)
        dec_attn: mean over heads
        Return: (H, F, F)
        """
        self.model.eval()

        running = None   # (H, F, F) accumulator (sum over batch)
        total_samples = 0
        all_labels = []
        for (X, y) in self.train_loader:
            X_input = X.to(self.device)
            loss, attn_enc, attn_self_list, attn_dec = \
                self.model(X_input, return_attn_weight=True)

            B, H, N, F = attn_enc.shape
            all_labels.append(y.cpu())
            
            if use_self_attn and attn_self_list:
                I = torch.eye(N, device=self.device).expand(B, H, N, N)  # (B,H,N,N)
                W_self_total = attn_self_list[0] + I
                for W in attn_self_list[1:]:
                    W_with_res = W + I
                    # (B,H,N,N) x (B,H,N,N) -> (B,H,N,N)
                    W_self_total = torch.einsum("bhij,bhjk->bhik", W_with_res, W_self_total)
                enc_mean = attn_enc.mean(dim=1)            # (B, N, F)
                dec_mean = attn_dec.mean(dim=1)            # (B, F, N)

                dec_mean_h = dec_mean.unsqueeze(1).expand(B, H, F, N)
                enc_mean_h = enc_mean.unsqueeze(1).expand(B, H, N, F)
            
            else:
                # no self attention -> dec x enc per each ehad
                W_self_total = torch.eye(N, device=self.device).expand(B, H, N, N)
                enc_mean_h = attn_enc
                dec_mean_h = attn_dec

            tmp = torch.einsum("bhfn,bhnk->bhfk", dec_mean_h, W_self_total)
            attn_feat_feat = torch.einsum("bhfn,bhnk->bhfk", tmp, enc_mean_h)

            if running is None:
                running = attn_feat_feat.sum(dim=0)  # (H, F, F)
            else:
                running += attn_feat_feat.sum(dim=0)

            total_samples += B

        result = running / total_samples  # (H, F, F)
        labels = torch.cat(all_labels, dim=0) 
        return result, labels


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
    def plot_reconstruction(self):
        """
        This method is for image dataset.
        """
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

        base_path = self.train_config['base_path']
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
        bins: int = 100,
        remove_outliers: bool = False,
        outlier_method: str = "percentile", # percentile or iqr
        low: float = 0.0,
        high: float = 100,
        iqr_k: float = 2.0,
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
            train_plot = train_normal
            train_plot = _clip(train_normal)
            test_norm_plot = test_normal
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


    def plot_attn_single(self):
        self.model.eval()

        test_it = iter(self.test_loader)
        for (X, y) in self.test_loader:
            X = X.to(self.device)
            loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = self.model(X, return_attn_weight=True)
            break
        single_attn_weight_enc = attn_weight_enc[0, :, :, :] # (H, F, N)
        single_attn_weight_dec = attn_weight_dec[0, :, :, :] # (H, F, N)

        depth = self.model.depth
        fig, axes = plt.subplots(1, depth + 2, figsize=(4 * (depth + 2), 4))

        single_attn_weight_enc_head_sum = single_attn_weight_enc.mean(dim=0) # (F, N)
        single_attn_weight_enc_head_sum = single_attn_weight_enc_head_sum.detach().cpu().numpy()  # (F, N)
        im_sum = axes[0].imshow(single_attn_weight_enc_head_sum, cmap='viridis', aspect='auto')
        axes[0].set_title('Attn Map of Encoder')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Latent')
        plt.colorbar(im_sum, ax=axes[0])

        for i, self_attn in enumerate(attn_weight_self_list):
            single_self_attn = self_attn[0] # get one sample            
            self_attn = single_self_attn.mean(0) # (F, N)
            
            head_sum_data = self_attn.detach().cpu().numpy()  # (F, N)
            im_sum = axes[i+1].imshow(head_sum_data, cmap='viridis', aspect='auto')
            axes[i+1].set_title(f'Heads Average of Self Attn {i+1}')
            axes[i+1].set_xlabel('Latent')
            axes[i+1].set_ylabel('Latent')
            plt.colorbar(im_sum, ax=axes[i+1])

        single_attn_weight_dec_head_sum = single_attn_weight_dec.mean(dim=0) # (F, N)
        single_attn_weight_dec_head_sum = single_attn_weight_dec_head_sum.detach().cpu().numpy()  # (F, N)
        im_sum = axes[5].imshow(single_attn_weight_dec_head_sum, cmap='viridis', aspect='auto')
        axes[5].set_title('Attn Map of Encoder')
        axes[5].set_xlabel('Column')
        axes[5].set_ylabel('Latent')
        plt.colorbar(im_sum, ax=axes[5])

        plt.tight_layout()
        filename = 'single_sample_attn_weight'
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f'{filename}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"Head-wise attention analysis saved to '{out_path}'")
        
        return


    def plot_attention_flexible(
        self,
        head_mode: str = 'sum',  # 'sum' or 'separate'
        self_depth_mode: str = 'sum',  # 'sum' or 'separate' 
        sample_idx: int = 0,
    ):
        """
        Flexible attention plotting with various options
        Plots normal and abnormal samples separately and saves them as individual files
        
        Args:
            head_mode: 'sum' (average heads) or 'separate' (plot each head)
            self_depth_mode: 'sum' (average depths) or 'separate' (plot each depth)
            sample_idx: which sample to analyze for each type (default: 0)
        """
        self.model.eval()

        def find_sample_by_label(loader, target_label, sample_idx=0):
            """Find a specific sample with target_label from loader"""
            samples_found = 0
            for (X, y) in loader:
                mask = (y == target_label)
                if mask.any():
                    target_samples = X[mask]
                    target_labels = y[mask]
                    if samples_found + target_samples.shape[0] > sample_idx:
                        relative_idx = sample_idx - samples_found
                        return target_samples[relative_idx:relative_idx+1], target_labels[relative_idx:relative_idx+1]
                    samples_found += target_samples.shape[0]
            return None, None

        def plot_single_sample(X_batch, y_batch, sample_type_name):
            """Plot attention for a single sample and return the plot path"""
            X_batch = X_batch.to(self.device)
            loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = self.model(X_batch, return_attn_weight=True)
            
            # Select first (and only) sample from batch
            single_attn_weight_enc = attn_weight_enc[0, :, :, :]  # (H, F, N)
            single_attn_weight_dec = attn_weight_dec[0, :, :, :]  # (H, F, N)
            single_attn_weight_self_list = [self_attn[0] for self_attn in attn_weight_self_list]  # List of (H, N, N)
            label = y_batch[0].item()
            
            # Determine subplot layout
            num_enc_plots = 1 if head_mode == 'sum' else single_attn_weight_enc.shape[0]  # H heads
            num_dec_plots = 1 if head_mode == 'sum' else single_attn_weight_dec.shape[0]  # H heads
            
            if self_depth_mode == 'sum':
                num_self_plots = 1 if head_mode == 'sum' else single_attn_weight_enc.shape[0]  # H heads
            else:  # separate depths
                num_self_plots = len(single_attn_weight_self_list) if head_mode == 'sum' else len(single_attn_weight_self_list) * single_attn_weight_enc.shape[0]
            
            total_plots = num_enc_plots + num_self_plots + num_dec_plots
            
            # Create subplots
            cols = min(total_plots, 6)  # Max 6 columns
            rows = (total_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            
            # Ensure axes is always 2D
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            plot_idx = 0
            
            # 1. Encoder Attention
            if head_mode == 'sum':
                enc_data = single_attn_weight_enc.mean(dim=0).detach().cpu().numpy()  # (F, N)
                ax = axes[plot_idx // cols, plot_idx % cols]
                im = ax.imshow(enc_data, cmap='viridis', aspect='auto')
                ax.set_title(f'{sample_type_name} Encoder (Heads Avg)\nLabel: {label}')
                ax.set_xlabel('Latent')
                ax.set_ylabel('Feature')
                plt.colorbar(im, ax=ax)
                plot_idx += 1
            else:
                for head_idx in range(single_attn_weight_enc.shape[0]):
                    enc_data = single_attn_weight_enc[head_idx].detach().cpu().numpy()  # (F, N)
                    ax = axes[plot_idx // cols, plot_idx % cols]
                    im = ax.imshow(enc_data, cmap='viridis', aspect='auto')
                    title_suffix = f'\nLabel: {label}' if head_idx == 0 else ''
                    ax.set_title(f'{sample_type_name} Encoder H{head_idx}{title_suffix}')
                    ax.set_xlabel('Latent')
                    ax.set_ylabel('Feature')
                    plt.colorbar(im, ax=ax)
                    plot_idx += 1
            
            # 2. Self Attention
            if self_depth_mode == 'sum':
                all_self_attn = torch.stack(single_attn_weight_self_list, dim=0)  # (Depth, H, N, N)
                depth_averaged = all_self_attn.mean(dim=0)  # (H, N, N)
                
                if head_mode == 'sum':
                    self_data = depth_averaged.mean(dim=0).detach().cpu().numpy()  # (N, N)
                    ax = axes[plot_idx // cols, plot_idx % cols]
                    im = ax.imshow(self_data, cmap='viridis', aspect='auto')
                    ax.set_title(f'{sample_type_name} Self (D+H Avg)')
                    ax.set_xlabel('Latent')
                    ax.set_ylabel('Latent')
                    plt.colorbar(im, ax=ax)
                    plot_idx += 1
                else:
                    for head_idx in range(depth_averaged.shape[0]):
                        self_data = depth_averaged[head_idx].detach().cpu().numpy()
                        ax = axes[plot_idx // cols, plot_idx % cols]
                        im = ax.imshow(self_data, cmap='viridis', aspect='auto')
                        ax.set_title(f'{sample_type_name} Self H{head_idx} (D Avg)')
                        ax.set_xlabel('Latent')
                        ax.set_ylabel('Latent')
                        plt.colorbar(im, ax=ax)
                        plot_idx += 1
            else:
                for depth_idx, self_attn in enumerate(single_attn_weight_self_list):
                    if head_mode == 'sum':
                        self_data = self_attn.mean(dim=0).detach().cpu().numpy()
                        ax = axes[plot_idx // cols, plot_idx % cols]
                        im = ax.imshow(self_data, cmap='viridis', aspect='auto')
                        ax.set_title(f'{sample_type_name} Self D{depth_idx} (H Avg)')
                        ax.set_xlabel('Latent')
                        ax.set_ylabel('Latent')
                        plt.colorbar(im, ax=ax)
                        plot_idx += 1
                    else:
                        for head_idx in range(self_attn.shape[0]):
                            self_data = self_attn[head_idx].detach().cpu().numpy()
                            ax = axes[plot_idx // cols, plot_idx % cols]
                            im = ax.imshow(self_data, cmap='viridis', aspect='auto')
                            ax.set_title(f'{sample_type_name} Self D{depth_idx}H{head_idx}')
                            ax.set_xlabel('Latent')
                            ax.set_ylabel('Latent')
                            plt.colorbar(im, ax=ax)
                            plot_idx += 1
            
            # 3. Decoder Attention
            if head_mode == 'sum':
                dec_data = single_attn_weight_dec.mean(dim=0).detach().cpu().numpy()
                ax = axes[plot_idx // cols, plot_idx % cols]
                im = ax.imshow(dec_data, cmap='viridis', aspect='auto')
                ax.set_title(f'{sample_type_name} Decoder (Heads Avg)')
                ax.set_xlabel('Latent')
                ax.set_ylabel('Feature')
                plt.colorbar(im, ax=ax)
                plot_idx += 1
            else:
                for head_idx in range(single_attn_weight_dec.shape[0]):
                    dec_data = single_attn_weight_dec[head_idx].detach().cpu().numpy()
                    ax = axes[plot_idx // cols, plot_idx % cols]
                    im = ax.imshow(dec_data, cmap='viridis', aspect='auto')
                    ax.set_title(f'{sample_type_name} Decoder H{head_idx}')
                    ax.set_xlabel('Latent')
                    ax.set_ylabel('Feature')
                    plt.colorbar(im, ax=ax)
                    plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, rows * cols):
                axes[idx // cols, idx % cols].axis('off')
            
            plt.tight_layout()
            
            # Generate filename
            filename_parts = ['attention', sample_type_name.lower(), f'idx{sample_idx}']
            if head_mode == 'separate':
                filename_parts.append('heads_sep')
            if self_depth_mode == 'separate':
                filename_parts.append('depths_sep')
            
            filename = '_'.join(filename_parts)
            base_path = self.train_config['base_path']
            out_path = os.path.join(base_path, f'{filename}.png')
            plt.savefig(out_path, bbox_inches='tight', dpi=200)
            plt.close()
            
            return out_path, label

        # Find normal and abnormal samples
        X_normal, y_normal = find_sample_by_label(self.test_loader, target_label=0, sample_idx=sample_idx)
        X_abnormal, y_abnormal = find_sample_by_label(self.test_loader, target_label=1, sample_idx=sample_idx)
        
        saved_paths = []
        
        # Plot normal sample
        if X_normal is not None:
            normal_path, normal_label = plot_single_sample(X_normal, y_normal, 'Normal')
            saved_paths.append(('Normal', normal_path, normal_label))
            print(f"Normal sample attention analysis saved to '{normal_path}' (label={normal_label})")
        else:
            print(f"Warning: Could not find normal sample at index {sample_idx}")
        
        # Plot abnormal sample
        if X_abnormal is not None:
            abnormal_path, abnormal_label = plot_single_sample(X_abnormal, y_abnormal, 'Abnormal')
            saved_paths.append(('Abnormal', abnormal_path, abnormal_label))
            print(f"Abnormal sample attention analysis saved to '{abnormal_path}' (label={abnormal_label})")
        else:
            print(f"Warning: Could not find abnormal sample at index {sample_idx}")
        
        print(f"Configuration: head_mode={head_mode}, self_depth_mode={self_depth_mode}")
        
        return saved_paths
    
    def plot_attn_simple(self, sample_idx: int = 0):
        """Simple 1x3 plot (original functionality)"""
        return self.plot_attention_flexible(
            head_mode='sum', 
            self_depth_mode='sum',
            sample_idx=sample_idx
        )

    def plot_attn_all_heads(self, sample_idx: int = 0):
        """Plot all heads separately but sum depths"""
        return self.plot_attention_flexible(
            head_mode='separate', 
            self_depth_mode='sum',
            sample_idx=sample_idx
        )

    def plot_attn_all_depths(self, sample_idx: int = 0):
        """Plot all depths separately but sum heads"""
        return self.plot_attention_flexible(
            head_mode='sum', 
            self_depth_mode='separate',
            sample_idx=sample_idx
        )

    def plot_attn_everything(self, sample_idx: int = 0):
        """Plot everything separately (heads AND depths)"""
        return self.plot_attention_flexible(
            head_mode='separate', 
            self_depth_mode='separate',
            sample_idx=sample_idx
        )

    @torch.no_grad()
    def plot_feature_reconstruction_distribution(
        self, 
        feature_idx: int = 0,
        bins: int = 50,
        alpha_hist: float = 0.6,
        alpha_kde: float = 0.8,
        figsize: tuple = (15, 5)
    ):
        """
        Plot KDE and histogram of a specific feature before/after reconstruction
        for normal and abnormal test samples
        
        Args:
            feature_idx: Index of the feature to analyze
            bins: Number of bins for histogram
            alpha_hist: Transparency for histogram
            alpha_kde: Transparency for KDE lines
            figsize: Figure size (width, height)
        """
        self.model.eval()
        
        def collect_feature_data(loader):
            """Collect original and reconstructed data for a specific feature"""
            original_data = []
            recon_data = []
            labels = []
            
            for (X, y) in loader:
                X_input = X.to(self.device)
                _, x_orig, x_recon = self.model(X_input, return_pred=True)
                
                original_data.append(x_orig[:, feature_idx].cpu().numpy())
                recon_data.append(x_recon[:, feature_idx].cpu().numpy())
                labels.append(y.cpu().numpy())
            
            original_data = np.concatenate(original_data, axis=0)
            recon_data = np.concatenate(recon_data, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            return original_data, recon_data, labels
        
        # Collect data from train and test loaders
        train_orig, train_recon, train_labels = collect_feature_data(self.train_loader)
        test_orig, test_recon, test_labels = collect_feature_data(self.test_loader)
        
        # Separate normal and abnormal test data
        test_normal_mask = (test_labels == 0)
        test_abnormal_mask = (test_labels == 1)
        
        test_normal_orig = test_orig[test_normal_mask]
        test_normal_recon = test_recon[test_normal_mask]
        test_abnormal_orig = test_orig[test_abnormal_mask] 
        test_abnormal_recon = test_recon[test_abnormal_mask]
        
        # Train data (only normal samples)
        train_normal_mask = (train_labels == 0)
        train_normal_data = train_orig[train_normal_mask]
        
        # Create subplots: Normal (left) and Abnormal (right)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)
        
        # Determine global x-axis limits for consistency
        all_data = np.concatenate([
            train_normal_data, 
            test_normal_orig, test_normal_recon,
            test_abnormal_orig, test_abnormal_recon
        ])
        x_min, x_max = all_data.min(), all_data.max()
        x_range = x_max - x_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        
        # Plot Normal samples (left)
        ax = axes[0]
        
        # Histogram
        ax.hist(train_normal_data, bins=bins, alpha=alpha_hist, density=True, 
                color='blue', label='Train (Normal)', range=(x_min, x_max))
        ax.hist(test_normal_orig, bins=bins, alpha=alpha_hist, density=True, 
                color='red', label='Test Normal (Original)', range=(x_min, x_max))
        ax.hist(test_normal_recon, bins=bins, alpha=alpha_hist, density=True, 
                color='green', label='Test Normal (Reconstructed)', range=(x_min, x_max))
        
        # KDE
        try:
            from scipy.stats import gaussian_kde
            
            if len(train_normal_data) > 1:
                kde_train = gaussian_kde(train_normal_data)
                x_range_plot = np.linspace(x_min, x_max, 200)
                ax.plot(x_range_plot, kde_train(x_range_plot), 
                    color='blue', linewidth=2, alpha=alpha_kde, label='Train KDE')
            
            if len(test_normal_orig) > 1:
                kde_test_orig = gaussian_kde(test_normal_orig)
                ax.plot(x_range_plot, kde_test_orig(x_range_plot), 
                    color='red', linewidth=2, alpha=alpha_kde, label='Test Normal Orig KDE')
            
            if len(test_normal_recon) > 1:
                kde_test_recon = gaussian_kde(test_normal_recon)
                ax.plot(x_range_plot, kde_test_recon(x_range_plot), 
                    color='green', linewidth=2, alpha=alpha_kde, linestyle='--', 
                    label='Test Normal Recon KDE')
                    
        except ImportError:
            print("Warning: scipy not available for KDE plotting")
        
        ax.set_title(f'Normal Samples - Feature {feature_idx}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        
        # Plot Abnormal samples (right)
        ax = axes[1]
        
        # Histogram
        ax.hist(train_normal_data, bins=bins, alpha=alpha_hist, density=True, 
                color='blue', label='Train (Normal)', range=(x_min, x_max))
        
        if len(test_abnormal_orig) > 0:
            ax.hist(test_abnormal_orig, bins=bins, alpha=alpha_hist, density=True, 
                    color='red', label='Test Abnormal (Original)', range=(x_min, x_max))
        if len(test_abnormal_recon) > 0:
            ax.hist(test_abnormal_recon, bins=bins, alpha=alpha_hist, density=True, 
                    color='green', label='Test Abnormal (Reconstructed)', range=(x_min, x_max))
        
        # KDE
        try:
            if len(train_normal_data) > 1:
                kde_train = gaussian_kde(train_normal_data)
                ax.plot(x_range_plot, kde_train(x_range_plot), 
                    color='blue', linewidth=2, alpha=alpha_kde, label='Train KDE')
            
            if len(test_abnormal_orig) > 1:
                kde_test_orig = gaussian_kde(test_abnormal_orig)
                ax.plot(x_range_plot, kde_test_orig(x_range_plot), 
                    color='red', linewidth=2, alpha=alpha_kde, label='Test Abnormal Orig KDE')
            
            if len(test_abnormal_recon) > 1:
                kde_test_recon = gaussian_kde(test_abnormal_recon)
                ax.plot(x_range_plot, kde_test_recon(x_range_plot), 
                    color='green', linewidth=2, alpha=alpha_kde, linestyle='--', 
                    label='Test Abnormal Recon KDE')
                    
        except ImportError:
            pass
        
        ax.set_title(f'Abnormal Samples - Feature {feature_idx}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        
        # Overall title
        fig.suptitle(f'Feature {feature_idx} Distribution: Before vs After Reconstruction • {self.train_config["dataset_name"].upper()}', 
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        # Save the plot
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f'feature_{feature_idx}_reconstruction_distribution.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        # Print statistics
        print(f"Feature {feature_idx} Statistics:")
        print(f"Train Normal - Mean: {train_normal_data.mean():.4f}, Std: {train_normal_data.std():.4f}")
        
        if len(test_normal_orig) > 0:
            print(f"Test Normal Original - Mean: {test_normal_orig.mean():.4f}, Std: {test_normal_orig.std():.4f}")
            print(f"Test Normal Reconstructed - Mean: {test_normal_recon.mean():.4f}, Std: {test_normal_recon.std():.4f}")
            print(f"Test Normal Reconstruction Shift - Mean: {abs(test_normal_orig.mean() - test_normal_recon.mean()):.4f}")
        
        if len(test_abnormal_orig) > 0:
            print(f"Test Abnormal Original - Mean: {test_abnormal_orig.mean():.4f}, Std: {test_abnormal_orig.std():.4f}")
            print(f"Test Abnormal Reconstructed - Mean: {test_abnormal_recon.mean():.4f}, Std: {test_abnormal_recon.std():.4f}")
            print(f"Test Abnormal Reconstruction Shift - Mean: {abs(test_abnormal_orig.mean() - test_abnormal_recon.mean()):.4f}")
            
            # Calculate shift towards training distribution
            train_mean = train_normal_data.mean()
            abnormal_orig_distance = abs(test_abnormal_orig.mean() - train_mean)
            abnormal_recon_distance = abs(test_abnormal_recon.mean() - train_mean)
            shift_towards_train = abnormal_orig_distance - abnormal_recon_distance
            print(f"Abnormal samples shift towards training mean: {shift_towards_train:.4f}")
        
        print(f"Feature distribution plot saved to '{out_path}'")
        
        return out_path

    def plot_2x3(self, sample_idx=0):
        """
        Create attention plot with normal average row and individual sample rows
        
        Args:
            sample_idx: int or list of ints for individual samples to analyze
        """
        self.model.eval()

        def find_sample_by_label(loader, target_label, sample_idx=0):
            samples_found = 0
            for (X, y) in loader:
                mask = (y == target_label)
                if mask.any():
                    target_samples = X[mask]
                    target_labels = y[mask]
                    if samples_found + target_samples.shape[0] > sample_idx:
                        relative_idx = sample_idx - samples_found
                        return target_samples[relative_idx:relative_idx+1], target_labels[relative_idx:relative_idx+1]
                    samples_found += target_samples.shape[0]
            return None, None

        def collect_all_normal_samples(loader):
            normal_samples = []
            normal_labels = []
            for (X, y) in loader:
                mask = (y == 0)
                if mask.any():
                    normal_samples.append(X[mask])
                    normal_labels.append(y[mask])
            
            if normal_samples:
                return torch.cat(normal_samples, dim=0), torch.cat(normal_labels, dim=0)
            return None, None

        def get_attention_maps(X_batch):
            X_batch = X_batch.to(self.device)
            loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = \
                self.model(X_batch, return_attn_weight=True)
            
            enc_attn = attn_weight_enc.mean(dim=(0, 1)).detach().cpu().numpy()
            dec_attn = attn_weight_dec.mean(dim=(0, 1)).detach().cpu().numpy()
            
            if attn_weight_self_list:
                all_self_attn = torch.stack([self_attn for self_attn in attn_weight_self_list], dim=0)
                self_attn = all_self_attn.mean(dim=(0, 1, 2)).detach().cpu().numpy()
            else:
                N = enc_attn.shape[1]
                self_attn = np.eye(N)
            
            return enc_attn, self_attn, dec_attn

        def get_single_sample_attention_maps(X_batch):
            X_batch = X_batch.to(self.device)
            loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = \
                self.model(X_batch, return_attn_weight=True)
            
            enc_attn = attn_weight_enc[0].mean(dim=0).detach().cpu().numpy()
            dec_attn = attn_weight_dec[0].mean(dim=0).detach().cpu().numpy()
            
            if attn_weight_self_list:
                all_self_attn = torch.stack([self_attn[0] for self_attn in attn_weight_self_list], dim=0)
                self_attn = all_self_attn.mean(dim=(0, 1)).detach().cpu().numpy()
            else:
                N = enc_attn.shape[1]
                self_attn = np.eye(N)
            
            return enc_attn, self_attn, dec_attn

        # Convert sample_idx to list
        if isinstance(sample_idx, int):
            sample_indices = [sample_idx]
        else:
            sample_indices = sample_idx

        # Collect normal samples for averaging
        X_normal_all, y_normal_all = collect_all_normal_samples(self.test_loader)
        if X_normal_all is None or len(X_normal_all) == 0:
            X_normal_all, y_normal_all = collect_all_normal_samples(self.train_loader)
        
        if X_normal_all is None:
            raise RuntimeError("Could not find normal samples for averaging")

        # Get normal average attention maps
        normal_avg_enc, normal_avg_self, normal_avg_dec = get_attention_maps(X_normal_all)

        # Collect individual samples
        individual_data = []
        for idx in sample_indices:
            X_single, y_single = find_sample_by_label(self.test_loader, target_label=0, sample_idx=idx)
            if X_single is None:
                X_single, y_single = find_sample_by_label(self.train_loader, target_label=0, sample_idx=idx)
            
            if X_single is None:
                raise RuntimeError(f"Could not find sample at index {idx}")
            
            single_enc, single_self, single_dec = get_single_sample_attention_maps(X_single)
            individual_data.append({
                'enc': single_enc,
                'self': single_self, 
                'dec': single_dec,
                'label': y_single[0].item(),
                'idx': idx
            })

        # Setup plot dimensions
        num_rows = 1 + len(sample_indices)
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), dpi=200)
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        # Calculate global color ranges
        all_cross_data = [normal_avg_enc, normal_avg_dec]
        all_self_data = [normal_avg_self]
        
        for data in individual_data:
            all_cross_data.extend([data['enc'], data['dec']])
            all_self_data.append(data['self'])
        
        cross_vmin, cross_vmax = min(d.min() for d in all_cross_data), max(d.max() for d in all_cross_data)
        self_vmin, self_vmax = min(d.min() for d in all_self_data), max(d.max() for d in all_self_data)

        # Plot normal average (row 0)
        normal_data = [normal_avg_enc, normal_avg_self, normal_avg_dec]
        normal_titles = ['Encoder Cross', 'Self', 'Decoder Cross']
        normal_xlabels = ['Latent', 'Latent', 'Latent']
        normal_ylabels = ['Feature', 'Latent', 'Feature']

        for col in range(3):
            if col == 1:  # Self attention
                vmin, vmax = self_vmin, self_vmax
            else:  # Cross attention
                vmin, vmax = cross_vmin, cross_vmax
            
            im = axes[0, col].imshow(normal_data[col], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, col].set_xlabel(normal_xlabels[col])
            axes[0, col].set_ylabel(normal_ylabels[col])
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Plot individual samples (remaining rows)
        for row_idx, data in enumerate(individual_data, 1):
            sample_data = [data['enc'], data['self'], data['dec']]
            
            for col in range(3):
                if col == 1:  # Self attention
                    vmin, vmax = self_vmin, self_vmax
                else:  # Cross attention
                    vmin, vmax = cross_vmin, cross_vmax
                
                im = axes[row_idx, col].imshow(sample_data[col], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                axes[row_idx, col].set_xlabel(normal_xlabels[col])
                axes[row_idx, col].set_ylabel(normal_ylabels[col])
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
                plt.colorbar(im, ax=axes[row_idx, col], fraction=0.046, pad=0.04)

        # Set title with more spacing
        sample_str = str(sample_indices) if len(sample_indices) > 1 else str(sample_indices[0])
        # fig.suptitle(f'Attention Maps: Normal Average vs Samples {sample_str} • {self.train_config["dataset_name"].upper()}', 
        #             fontsize=16, y=0.98)

        plt.tight_layout()

        # Save plot
        base_path = self.train_config['base_path']
        sample_filename = "_".join(map(str, sample_indices)) if len(sample_indices) > 1 else str(sample_indices[0])
        out_path = os.path.join(base_path, f'attention_{num_rows}x3_samples_{sample_filename}_{self.train_config["dataset_name"]}.pdf')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        out_path = os.path.join(base_path, f'attention_{num_rows}x3_samples_{sample_filename}_{self.train_config["dataset_name"]}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"{num_rows}x3 attention plot saved to '{out_path}'")
        print(f"Normal samples averaged: {len(X_normal_all)}")
        for data in individual_data:
            print(f"Sample {data['idx']} label: {data['label']}")
        
        return out_path
    @torch.no_grad()
    def plot_2x4(self, abnormal_idx=0, abnormal_avg=False, plot_heads=False, plot_2x2=False):
        """
        Create a 2x4 plot showing:
        Row 1: Normal Enc Map, Normal Self Map, Normal Z Dec Map, Normal Z_hat Dec Map
        Row 2: Abnormal Enc Map, Abnormal Self Map, Abnormal Z Dec Map, Abnormal Z_hat Dec Map
        
        If plot_2x2=True, create a 2x2 plot showing only:
        Row 1: Normal Z Dec Map, Normal Z_hat Dec Map
        Row 2: Abnormal Z Dec Map, Abnormal Z_hat Dec Map
        
        Args:
            abnormal_idx: which abnormal sample to analyze (ignored if abnormal_avg=True)
            abnormal_avg: if True, average all abnormal samples; if False, use specific sample
            plot_heads: if True, create separate plots for each head (assumes 4 heads)
            plot_2x2: if True, create 2x2 decoder-only plot; if False, create 2x4 full plot
        """
        self.model.eval()

        def find_sample_by_label(loader, target_label, sample_idx=0):
            """Find a specific sample with target_label from loader"""
            samples_found = 0
            for (X, y) in loader:
                mask = (y == target_label)
                if mask.any():
                    target_samples = X[mask]
                    target_labels = y[mask]
                    if samples_found + target_samples.shape[0] > sample_idx:
                        relative_idx = sample_idx - samples_found
                        return target_samples[relative_idx:relative_idx+1], target_labels[relative_idx:relative_idx+1]
                    samples_found += target_samples.shape[0]
            return None, None

        def collect_all_abnormal_samples(loader):
            """Collect all abnormal samples from loader"""
            abnormal_samples = []
            abnormal_labels = []
            for (X, y) in loader:
                mask = (y != 0)
                if mask.any():
                    abnormal_samples.append(X[mask])
                    abnormal_labels.append(y[mask])
            
            if abnormal_samples:
                return torch.cat(abnormal_samples, dim=0), torch.cat(abnormal_labels, dim=0)
            return None, None

        def get_attention_maps_and_decoder_variants(X_batch):
            """Get encoder, self, and both decoder attention maps (z and z_hat versions)"""
            X_batch = X_batch.to(self.device)
            
            # Get all analysis data including both decoder attentions
            loss, x, x_hat, latents, latents_hat, memory_weight, attn_weight_enc, attn_weight_self_list, attn_weight_dec_z, attn_weight_dec_z_hat = \
                self.model(X_batch, return_for_analysis=True)
            
            if plot_heads:
                # Keep head dimension for head-wise plotting (assumes 4 heads)
                # Encoder: (B, H, F, N) -> average over batch -> (H, F, N)
                enc_attn = attn_weight_enc.mean(dim=0).detach().cpu().numpy()  # (H=4, F, N)
                # Decoder Z: (B, H, F, N) -> average over batch -> (H, F, N)
                dec_attn_z = attn_weight_dec_z.mean(dim=0).detach().cpu().numpy()  # (H=4, F, N)
                # Decoder Z_hat: (B, H, F, N) -> average over batch -> (H, F, N)
                dec_attn_z_hat = attn_weight_dec_z_hat.mean(dim=0).detach().cpu().numpy()  # (H=4, F, N)
                
                # Self attention: average over layers but keep heads
                if attn_weight_self_list:
                    all_self_attn = torch.stack([self_attn for self_attn in attn_weight_self_list], dim=0)  # (Depth, B, H, N, N)
                    self_attn = all_self_attn.mean(dim=(0, 1)).detach().cpu().numpy()  # (H=4, N, N)
                else:
                    N = enc_attn.shape[2]  # Note: shape is now (H, F, N)
                    self_attn = np.tile(np.eye(N)[None, :, :], (4, 1, 1))  # (H=4, N, N)
                    
            else:
                # Average over heads and samples (original behavior)
                enc_attn = attn_weight_enc.mean(dim=(0, 1)).detach().cpu().numpy()  # (F, N)
                dec_attn_z = attn_weight_dec_z.mean(dim=(0, 1)).detach().cpu().numpy()  # (F, N)
                dec_attn_z_hat = attn_weight_dec_z_hat.mean(dim=(0, 1)).detach().cpu().numpy()  # (F, N)
                
                # Average self attention over layers and heads
                if attn_weight_self_list:
                    all_self_attn = torch.stack([self_attn for self_attn in attn_weight_self_list], dim=0)  # (Depth, B, H, N, N)
                    self_attn = all_self_attn.mean(dim=(0, 1, 2)).detach().cpu().numpy()  # (N, N)
                else:
                    N = enc_attn.shape[1]
                    self_attn = np.eye(N)
            
            return enc_attn, self_attn, dec_attn_z, dec_attn_z_hat

        # Find normal sample (always use first normal sample for consistency)
        X_normal, y_normal = find_sample_by_label(self.test_loader, target_label=0, sample_idx=0)
        
        if X_normal is None:
            print("Warning: Could not find normal sample")
            return None
        
        # Get abnormal samples
        if abnormal_avg:
            X_abnormal, y_abnormal = collect_all_abnormal_samples(self.test_loader)
            if X_abnormal is None or len(X_abnormal) == 0:
                print("Warning: Could not find abnormal samples")
                return None
            abnormal_label_text = f"Abnormal (avg of {len(X_abnormal)} samples)"
        else:
            X_abnormal, y_abnormal = find_sample_by_label(self.test_loader, target_label=1, sample_idx=abnormal_idx)
            if X_abnormal is None:
                print(f"Warning: Could not find abnormal sample at index {abnormal_idx}")
                return None
            abnormal_label_text = f"Abnormal (sample {abnormal_idx}, label: {y_abnormal[0].item()})"
        
        # Get attention maps for both normal and abnormal
        normal_enc, normal_self, normal_dec_z, normal_dec_z_hat = get_attention_maps_and_decoder_variants(X_normal)
        abnormal_enc, abnormal_self, abnormal_dec_z, abnormal_dec_z_hat = get_attention_maps_and_decoder_variants(X_abnormal)
        
        if plot_2x2:
            # 2x2 plot: only decoder attention maps
            if plot_heads:
                # Create separate plots for each head
                saved_paths = []
                
                for head_idx in range(4):  # Assuming 4 heads
                    # Create 2x2 subplot for this head
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200)
                    
                    # Extract data for current head - only decoder maps
                    plots_config = [
                        # Row 0: Normal samples
                        (normal_dec_z[head_idx], f'Before Addressing', 'Latent Index', 'Feature Index'),
                        (normal_dec_z_hat[head_idx], f'After Addressing', 'Latent Index', 'Feature Index'),
                        
                        (abnormal_dec_z[head_idx], f' ', 'Latent Index', 'Feature Index'),
                        (abnormal_dec_z_hat[head_idx], f' ', 'Latent Index', 'Feature Index')
                    ]
                    
                    # Find global min/max for each decoder type for this head
                    dec_z_data = [normal_dec_z[head_idx], abnormal_dec_z[head_idx]]
                    dec_z_hat_data = [normal_dec_z_hat[head_idx], abnormal_dec_z_hat[head_idx]]
                    
                    dec_z_vmin, dec_z_vmax = min(d.min() for d in dec_z_data), max(d.max() for d in dec_z_data)
                    dec_z_hat_vmin, dec_z_hat_vmax = min(d.min() for d in dec_z_hat_data), max(d.max() for d in dec_z_hat_data)
                    
                    v_ranges = [
                        (dec_z_vmin, dec_z_vmax),       # Column 0: Decoder Z
                        (dec_z_hat_vmin, dec_z_hat_vmax) # Column 1: Decoder Z_hat
                    ]
                    
                    # Plot each attention map
                    for idx, (attn_data, title, xlabel, ylabel) in enumerate(plots_config):
                        row, col = idx // 2, idx % 2
                        vmin, vmax = v_ranges[col]
                        
                        im = axes[row, col].imshow(attn_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                        axes[row, col].set_title(title, fontsize=12)
                        
                        # Remove tick labels for cleaner appearance
                        axes[row, col].set_xticks([])
                        axes[row, col].set_yticks([])
                        
                        # Add colorbar to each subplot
                        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                    
                    axes[0, 0].set_ylabel('Latent Index')
                    axes[1, 0].set_ylabel('Latent Index')
                    axes[1, 0].set_xlabel('Column Index')
                    axes[1, 1].set_xlabel('Column Index')

                    # Overall title for this head
                    title_suffix = " (Abnormal Averaged)" if abnormal_avg else f" (Abnormal Sample {abnormal_idx})"
                    fig.suptitle(f'Head {head_idx} Decoder Attention Maps: Normal vs Abnormal{title_suffix} • {self.train_config["dataset_name"].upper()}', 
                                fontsize=16, y=0.98)
                    
                    plt.tight_layout()
                    
                    # Save the plot for this head
                    base_path = self.train_config['base_path']
                    filename_suffix = "_avg" if abnormal_avg else f"_idx{abnormal_idx}"
                    out_path = os.path.join(base_path, f'attention_2x2_decoder_head{head_idx}{filename_suffix}.png')
                    plt.savefig(out_path, bbox_inches='tight', dpi=200)
                    plt.close()
                    
                    saved_paths.append(out_path)
                    print(f"Head {head_idx} decoder attention comparison saved to '{out_path}'")
                
                print(f"All {len(saved_paths)} head-wise 2x2 plots saved successfully")
                return saved_paths
                
            else:
                # Single 2x2 plot with averaged heads
                fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200)
                
                # Plot configurations: (data, title, xlabel, ylabel) - only decoder maps
                plots_config = [
                    # Row 0: Normal samples
                    (normal_dec_z, f'Before Addressing', 'Latent Index', 'Feature Index'),
                    (normal_dec_z_hat, 'After Addresing', 'Before Addressing', 'Feature Index'),
                    
                    # Row 1: Abnormal samples
                    (abnormal_dec_z, f'Abnormal Decoder (Z)\n({abnormal_label_text.split("(")[1][:-1]})', 'Latent Index', 'Feature Index'),
                    (abnormal_dec_z_hat, 'Abnormal Decoder (Z_hat)', 'Latent Index', 'Feature Index')
                ]
                
                # Find global min/max for each decoder type for consistent color scaling
                dec_z_data = [normal_dec_z, abnormal_dec_z]
                dec_z_hat_data = [normal_dec_z_hat, abnormal_dec_z_hat]
                
                dec_z_vmin, dec_z_vmax = min(d.min() for d in dec_z_data), max(d.max() for d in dec_z_data)
                dec_z_hat_vmin, dec_z_hat_vmax = min(d.min() for d in dec_z_hat_data), max(d.max() for d in dec_z_hat_data)
                
                v_ranges = [
                    (dec_z_vmin, dec_z_vmax),       # Column 0: Decoder Z
                    (dec_z_hat_vmin, dec_z_hat_vmax) # Column 1: Decoder Z_hat
                ]
                
                # Plot each attention map
                for idx, (attn_data, title, xlabel, ylabel) in enumerate(plots_config):
                    row, col = idx // 2, idx % 2
                    vmin, vmax = v_ranges[col]
                    
                    im = axes[row, col].imshow(attn_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                    # axes[row, col].set_title(title, fontsize=12)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    
                
                # title
                axes[0, 0].set_title('Before Addressing', fontsize=24, pad=20)
                axes[0, 1].set_title('After Addressing', fontsize=24, pad=20)
                axes[1, 0].set_title(' ', fontsize=24, pad=20)
                axes[1, 1].set_title(' ', fontsize=24, pad=20)
                
                # axis name
                axes[0, 0].set_ylabel('Column', fontsize=16, labelpad=15)
                axes[1, 0].set_ylabel('Column', fontsize=16, labelpad=15)
                axes[1, 0].set_xlabel('Latent', fontsize=16, labelpad=15)
                axes[1, 1].set_xlabel('Latent', fontsize=16, labelpad=15)


                # Overall title
                # title_suffix = " (Abnormal Averaged)" if abnormal_avg else f" (Abnormal Sample {abnormal_idx})"
                # fig.suptitle(f'Decoder Attention Maps: Normal vs Abnormal{title_suffix} • {self.train_config["dataset_name"].upper()}', 
                #             fontsize=16, y=0.98)
                
                plt.tight_layout()
                
                # Save the plot
                base_path = self.train_config['base_path']
                filename_suffix = "_avg" if abnormal_avg else f"_idx{abnormal_idx}"
                out_path = os.path.join(base_path, f'attention_2x2_decoder{filename_suffix}_{self.train_config["dataset_name"]}.pdf')
                plt.savefig(out_path, bbox_inches='tight', dpi=200)
                out_path = os.path.join(base_path, f'attention_2x2_decoder{filename_suffix}_{self.train_config["dataset_name"]}.png')
                plt.savefig(out_path, bbox_inches='tight', dpi=200)
                plt.close()
                print(f"2x2 decoder attention comparison saved to '{out_path}'")
                
                return out_path
        
        # Original 2x4 plot code continues here...
        if plot_heads:
            # Create separate plots for each head
            saved_paths = []
            
            for head_idx in range(4):  # Assuming 4 heads
                # Create 2x4 subplot for this head
                fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=200)
                
                # Extract data for current head
                plots_config = [
                    # Row 0: Normal samples
                    (normal_enc[head_idx], f'Normal Encoder H{head_idx}\n(Label: {y_normal[0].item()})', 'Latent Index', 'Feature Index'),
                    (normal_self[head_idx], f'Normal Self-Attention H{head_idx}', 'Latent Index', 'Latent Index'),
                    (normal_dec_z[head_idx], f'Normal Decoder H{head_idx} (Z)', 'Latent Index', 'Feature Index'),
                    (normal_dec_z_hat[head_idx], f'Normal Decoder H{head_idx} (Z_hat)', 'Latent Index', 'Feature Index'),
                    
                    # Row 1: Abnormal samples
                    (abnormal_enc[head_idx], f'Abnormal Encoder H{head_idx}\n({abnormal_label_text.split("(")[1][:-1]})', 'Latent Index', 'Feature Index'),
                    (abnormal_self[head_idx], f'Abnormal Self-Attention H{head_idx}', 'Latent Index', 'Latent Index'),
                    (abnormal_dec_z[head_idx], f'Abnormal Decoder H{head_idx} (Z)', 'Latent Index', 'Feature Index'),
                    (abnormal_dec_z_hat[head_idx], f'Abnormal Decoder H{head_idx} (Z_hat)', 'Latent Index', 'Feature Index')
                ]
                
                # Find global min/max for each attention type for this head
                enc_data = [normal_enc[head_idx], abnormal_enc[head_idx]]
                self_data = [normal_self[head_idx], abnormal_self[head_idx]]
                dec_z_data = [normal_dec_z[head_idx], abnormal_dec_z[head_idx]]
                dec_z_hat_data = [normal_dec_z_hat[head_idx], abnormal_dec_z_hat[head_idx]]
                
                enc_vmin, enc_vmax = min(d.min() for d in enc_data), max(d.max() for d in enc_data)
                self_vmin, self_vmax = min(d.min() for d in self_data), max(d.max() for d in self_data)
                dec_z_vmin, dec_z_vmax = min(d.min() for d in dec_z_data), max(d.max() for d in dec_z_data)
                dec_z_hat_vmin, dec_z_hat_vmax = min(d.min() for d in dec_z_hat_data), max(d.max() for d in dec_z_hat_data)
                
                v_ranges = [
                    (enc_vmin, enc_vmax),           # Column 0: Encoder
                    (self_vmin, self_vmax),         # Column 1: Self-attention  
                    (dec_z_vmin, dec_z_vmax),       # Column 2: Decoder Z
                    (dec_z_hat_vmin, dec_z_hat_vmax) # Column 3: Decoder Z_hat
                ]
                
                # Plot each attention map
                for idx, (attn_data, title, xlabel, ylabel) in enumerate(plots_config):
                    row, col = idx // 4, idx % 4
                    vmin, vmax = v_ranges[col]
                    
                    im = axes[row, col].imshow(attn_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                    axes[row, col].set_title(title, fontsize=12)
                    axes[row, col].set_xlabel(xlabel)
                    axes[row, col].set_ylabel(ylabel)
                    
                    # Remove tick labels for cleaner appearance
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    
                    # Add colorbar to each subplot
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                
                # Overall title for this head
                title_suffix = " (Abnormal Averaged)" if abnormal_avg else f" (Abnormal Sample {abnormal_idx})"
                fig.suptitle(f'Head {head_idx} Attention Maps: Normal vs Abnormal{title_suffix} • {self.train_config["dataset_name"].upper()}', 
                            fontsize=16, y=0.98)
                
                plt.tight_layout()
                
                # Save the plot for this head
                base_path = self.train_config['base_path']
                filename_suffix = "_avg" if abnormal_avg else f"_idx{abnormal_idx}"
                out_path = os.path.join(base_path, f'attention_2x4_comparison_head{head_idx}{filename_suffix}.png')
                plt.savefig(out_path, bbox_inches='tight', dpi=200)
                plt.close()
                
                saved_paths.append(out_path)
                print(f"Head {head_idx} attention comparison saved to '{out_path}'")
            
            print(f"All {len(saved_paths)} head-wise plots saved successfully")
            if abnormal_avg:
                print(f"Normal sample label: {y_normal[0].item()}, Abnormal samples: averaged over {len(X_abnormal)} samples")
            else:
                print(f"Normal sample label: {y_normal[0].item()}, Abnormal sample label: {y_abnormal[0].item()}")
            
            return saved_paths
            
        else:
            # Original behavior: single plot with averaged heads
            fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=200)
            
            # Plot configurations: (data, title, xlabel, ylabel)
            plots_config = [
                # Row 0: Normal samples
                (normal_enc, f'Normal Encoder\n(Label: {y_normal[0].item()})', 'Latent Index', 'Feature Index'),
                (normal_self, 'Normal Self-Attention', 'Latent Index', 'Latent Index'),
                (normal_dec_z, 'Normal Decoder (Z)', 'Latent Index', 'Feature Index'),
                (normal_dec_z_hat, 'Normal Decoder (Z_hat)', 'Latent Index', 'Feature Index'),
                
                # Row 1: Abnormal samples
                (abnormal_enc, f'Abnormal Encoder\n({abnormal_label_text.split("(")[1][:-1]})', 'Latent Index', 'Feature Index'),
                (abnormal_self, 'Abnormal Self-Attention', 'Latent Index', 'Latent Index'),
                (abnormal_dec_z, 'Abnormal Decoder (Z)', 'Latent Index', 'Feature Index'),
                (abnormal_dec_z_hat, 'Abnormal Decoder (Z_hat)', 'Latent Index', 'Feature Index')
            ]
            
            # Find global min/max for each attention type for consistent color scaling
            enc_data = [normal_enc, abnormal_enc]
            self_data = [normal_self, abnormal_self]
            dec_z_data = [normal_dec_z, abnormal_dec_z]
            dec_z_hat_data = [normal_dec_z_hat, abnormal_dec_z_hat]
            
            enc_vmin, enc_vmax = min(d.min() for d in enc_data), max(d.max() for d in enc_data)
            self_vmin, self_vmax = min(d.min() for d in self_data), max(d.max() for d in self_data)
            dec_z_vmin, dec_z_vmax = min(d.min() for d in dec_z_data), max(d.max() for d in dec_z_data)
            dec_z_hat_vmin, dec_z_hat_vmax = min(d.min() for d in dec_z_hat_data), max(d.max() for d in dec_z_hat_data)
            
            v_ranges = [
                (enc_vmin, enc_vmax),           # Column 0: Encoder
                (self_vmin, self_vmax),         # Column 1: Self-attention  
                (dec_z_vmin, dec_z_vmax),       # Column 2: Decoder Z
                (dec_z_hat_vmin, dec_z_hat_vmax) # Column 3: Decoder Z_hat
            ]
            
            # Plot each attention map
            for idx, (attn_data, title, xlabel, ylabel) in enumerate(plots_config):
                row, col = idx // 4, idx % 4
                vmin, vmax = v_ranges[col]
                
                im = axes[row, col].imshow(attn_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                axes[row, col].set_title(title, fontsize=12)
                axes[row, col].set_xlabel(xlabel)
                axes[row, col].set_ylabel(ylabel)
                
                # Remove tick labels for cleaner appearance
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                
                # Add colorbar to each subplot
                # plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            axes[0, 0].set_ylabel('Latent Index')
            axes[1, 0].set_ylabel('Latent Index')
            axes[1, 0].set_xlabel('Column Index')
            axes[1, 1].set_xlabel('Column Index')

            
            # Overall title
            title_suffix = " (Abnormal Averaged)" if abnormal_avg else f" (Abnormal Sample {abnormal_idx})"
            fig.suptitle(f'Attention Maps: Normal vs Abnormal{title_suffix} • {self.train_config["dataset_name"].upper()}', 
                        fontsize=16, y=0.98)
            
            plt.tight_layout()
            
            # Save the plot
            base_path = self.train_config['base_path']
            filename_suffix = "_avg" if abnormal_avg else f"_idx{abnormal_idx}"
            out_path = os.path.join(base_path, f'attention_2x4_comparison{filename_suffix}_{self.train_config["dataset_name"]}.pdf')
            plt.savefig(out_path, bbox_inches='tight', dpi=200)
            out_path = os.path.join(base_path, f'attention_2x4_comparison{filename_suffix}_{self.train_config["dataset_name"]}.png')
            plt.savefig(out_path, bbox_inches='tight', dpi=200)
            plt.close()
            print(f"2x4 attention comparison saved to '{out_path}'")

            if abnormal_avg:
                print(f"Normal sample label: {y_normal[0].item()}, Abnormal samples: averaged over {len(X_abnormal)} samples")
            else:
                print(f"Normal sample label: {y_normal[0].item()}, Abnormal sample label: {y_abnormal[0].item()}")
            
            return out_path



    @torch.no_grad()
    def find_most_different_abnormal_sample(self):
        """
        Find abnormal sample with maximum difference between decoder attention maps
        before and after memory addressing.
        
        Returns:
            best_idx: index of the abnormal sample with maximum attention difference
            best_difference: the maximum difference value found
        """
        self.model.eval()
        
        def compute_attention_difference(X_batch):
            """Compute difference between pre and post addressing decoder attention maps"""
            X_batch = X_batch.to(self.device)
            
            # Get decoder attention maps before and after addressing
            loss, x, x_hat, latents, latents_hat, memory_weight, attn_weight_enc, attn_weight_self_list, attn_weight_dec_z, attn_weight_dec_z_hat = \
                self.model(X_batch, return_for_analysis=True)
            
            # Average over heads: (B, H, F, N) -> (B, F, N)
            dec_z = attn_weight_dec_z.mean(dim=1).detach().cpu().numpy()
            dec_z_hat = attn_weight_dec_z_hat.mean(dim=1).detach().cpu().numpy()
            
            # Compute L2 difference for each sample in batch
            differences = []
            for i in range(dec_z.shape[0]):
                diff = np.linalg.norm(dec_z_hat[i] - dec_z[i])
                differences.append(diff)
            
            return np.array(differences)
        
        # Find abnormal samples and compute differences
        best_idx = -1
        best_difference = -1.0
        current_abnormal_idx = 0
        
        print("Searching for abnormal sample with maximum attention difference...")
        
        for batch_idx, (X, y) in enumerate(self.test_loader):
            # Find abnormal samples in this batch
            abnormal_mask = (y != 0)
            
            if not abnormal_mask.any():
                continue
                
            abnormal_X = X[abnormal_mask]
            abnormal_y = y[abnormal_mask]
            
            # Compute attention differences for abnormal samples
            differences = compute_attention_difference(abnormal_X)
            
            # Check each abnormal sample in this batch
            for i, diff in enumerate(differences):
                if diff > best_difference:
                    best_difference = diff
                    best_idx = current_abnormal_idx + i
                    print(f"New best abnormal sample found: idx={best_idx}, difference={diff:.4f}")
            
            current_abnormal_idx += len(differences)
        
        if best_idx == -1:
            print("Warning: No abnormal samples found!")
            return None, None
        
        print(f"\nBest abnormal sample: index={best_idx}, attention_difference={best_difference:.4f}")
        return best_idx, best_difference

    def analyze_best_abnormal_sample(self):
        """
        Find the abnormal sample with maximum attention difference and create 2x2 plot
        """
        # Find the best abnormal sample
        best_idx, best_diff = self.find_most_different_abnormal_sample()
        
        if best_idx is None:
            print("Cannot proceed: No abnormal samples found")
            return None
        
        print(f"\nCreating 2x2 plot for abnormal sample {best_idx} (difference: {best_diff:.4f})")
        
        # Create 2x2 plot with the best sample
        plot_path = self.plot_2x4(abnormal_idx=best_idx, abnormal_avg=False, plot_2x2=True)
        
        return {
            'best_abnormal_idx': best_idx,
            'attention_difference': best_diff,
            'plot_path': plot_path
        }



    @torch.no_grad()
    def visualize_attention_vs_shap(self):
        """
        Simple visualization comparing encoder attention map with SHAP values
        """
        import xgboost as xgb
        import shap
        
        print("Starting simple attention vs SHAP visualization...")
        
        # 1. Get test data for XGBoost
        X_test, y_test = [], []
        for X, y in self.test_loader:
            X_test.append(X.cpu().numpy())
            y_test.append(y.cpu().numpy())
        
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Label distribution: Normal={np.sum(y_test==0)}, Abnormal={np.sum(y_test!=0)}")
        
        # 2. Train XGBoost and get SHAP values
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_test, y_test)
        
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        
        # Mean absolute SHAP values per feature
        shap_importance = np.mean(np.abs(shap_values), axis=0)  # Shape: (F,)
        print(f"SHAP importance shape: {shap_importance.shape}")
        
        # 3. Get encoder attention weights
        print("Getting encoder attention weights...")
        self.model.eval()
        
        all_enc_attn = []
        for X, y in self.test_loader:
            X = X.to(self.device)
            loss, attn_enc, attn_self_list, attn_dec = self.model(X, return_attn_weight=True)
            
            # attn_enc shape: (B, H, N, F)
            # Average over batch and heads: (N, F)
            enc_attn_avg = attn_enc.mean(dim=(0, 1)).detach().cpu().numpy()
            all_enc_attn.append(enc_attn_avg)
            break  # Just use first batch for visualization
        
        enc_attention = all_enc_attn[0]  # Shape: (N, F)
        print(f"Encoder attention shape: {enc_attention.shape}")
        
        N, F = enc_attention.shape
        print(f"N (latent dimensions): {N}, F (features): {F}")
        
        # Check if dimensions match
        if F != len(shap_importance):
            print(f"WARNING: Feature dimension mismatch!")
            print(f"SHAP features: {len(shap_importance)}, Attention features: {F}")
            min_features = min(len(shap_importance), F)
            shap_importance = shap_importance[:min_features]
            enc_attention = enc_attention[:, :min_features]
            print(f"Using first {min_features} features for comparison")
            F = min_features
        
        # 4. Create visualizations
        base_path = self.train_config['base_path']
        
        # Plot 1: SHAP importance bar plot
        plt.figure(figsize=(15, 5), dpi=200)
        feature_indices = np.arange(F)
        plt.bar(feature_indices, shap_importance, alpha=0.7, color='red')
        plt.title(f'SHAP Feature Importance • {self.train_config["dataset_name"].upper()}')
        plt.xlabel('Feature Index')
        plt.ylabel('Mean |SHAP Value|')
        plt.grid(True, alpha=0.3)
        
        # Highlight top 10 features
        top_10_indices = np.argsort(shap_importance)[-10:]
        plt.bar(top_10_indices, shap_importance[top_10_indices], alpha=0.9, color='darkred')
        
        plt.tight_layout()
        shap_path = os.path.join(base_path, 'shap_importance_bar.png')
        plt.savefig(shap_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"SHAP importance plot saved to {shap_path}")
        
        # Plot 2: Encoder attention heatmap
        plt.figure(figsize=(15, 8), dpi=200)
        im = plt.imshow(enc_attention, cmap='viridis', aspect='auto')
        plt.title(f'Encoder Attention Map • {self.train_config["dataset_name"].upper()}')
        plt.xlabel('Feature Index')
        plt.ylabel('Latent Index')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Add grid for better readability
        if F <= 50:  # Only add grid if not too many features
            plt.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        attn_path = os.path.join(base_path, 'encoder_attention_heatmap.png')
        plt.savefig(attn_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Encoder attention heatmap saved to {attn_path}")
        
        # Plot 3: Average attention per feature vs SHAP
        avg_attention_per_feature = np.mean(np.abs(enc_attention), axis=0)  # Average over latent dimensions
        
        plt.figure(figsize=(12, 8), dpi=200)
        
        # Subplot 1: Bar comparison
        plt.subplot(2, 1, 1)
        x = np.arange(F)
        width = 0.35
        
        # Normalize for fair comparison
        shap_norm = shap_importance / np.max(shap_importance)
        attn_norm = avg_attention_per_feature / np.max(avg_attention_per_feature)
        
        plt.bar(x - width/2, shap_norm, width, label='SHAP (normalized)', alpha=0.7, color='red')
        plt.bar(x + width/2, attn_norm, width, label='Attention (normalized)', alpha=0.7, color='blue')
        
        plt.xlabel('Feature Index')
        plt.ylabel('Normalized Importance')
        plt.title(f'SHAP vs Attention Importance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Scatter plot
        plt.subplot(2, 1, 2)
        plt.scatter(shap_norm, attn_norm, alpha=0.6, s=30)
        plt.xlabel('SHAP Importance (normalized)')
        plt.ylabel('Attention Importance (normalized)')
        plt.title('SHAP vs Attention Scatter Plot')
        
        # Calculate correlation
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(shap_norm, attn_norm)
        spearman_r, spearman_p = spearmanr(shap_norm, attn_norm)
        
        plt.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f} (p={pearson_p:.3f})\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.3f})', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(base_path, 'shap_vs_attention_comparison.png')
        plt.savefig(comparison_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Comparison plot saved to {comparison_path}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
        print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
        
        # Top features analysis
        top_k = 10
        top_shap_indices = set(np.argsort(shap_importance)[-top_k:])
        top_attn_indices = set(np.argsort(avg_attention_per_feature)[-top_k:])
        overlap = top_shap_indices.intersection(top_attn_indices)
        
        print(f"\nTop {top_k} features:")
        print(f"SHAP top features: {sorted(list(top_shap_indices))}")
        print(f"Attention top features: {sorted(list(top_attn_indices))}")
        print(f"Overlap: {len(overlap)}/{top_k} features: {sorted(list(overlap))}")
        
        return {
            'shap_importance': shap_importance,
            'attention_importance': avg_attention_per_feature,
            'correlations': {
                'pearson': pearson_r,
                'pearson_p': pearson_p,
                'spearman': spearman_r,
                'spearman_p': spearman_p
            },
            'top_feature_overlap': overlap,
            'plot_paths': {
                'shap_bar': shap_path,
                'attention_heatmap': attn_path,
                'comparison': comparison_path
            }
        }

    @torch.no_grad()
    def compare_shap_vs_encoder_attention_per_sample(
        self, 
        similarity_threshold: float = 0.7,
        top_k_samples: int = 10,
        figsize: tuple = (12, 6),
        plot_group_average: bool = True,
        include_self_attention: bool = False
    ):
        """
        Compare XGBoost SHAP values with MemPAE encoder attention maps for individual abnormal samples.
        Saves 1x2 plots for samples where SHAP and attention patterns are similar.
        
        Args:
            similarity_threshold: Minimum correlation to consider patterns "similar"
            top_k_samples: Number of most similar samples to save plots for
            figsize: Figure size for each 1x2 plot
            plot_group_average: If True, also plot average SHAP vs attention for normal/abnormal groups
            include_self_attention: If True, include self-attention in the attention computation (enc @ self_attn_1 @ ... @ self_attn_L)
            
        Returns:
            dict: Results including correlations, saved plots info, and statistics
        """
        import xgboost as xgb
        import shap
        from scipy.stats import pearsonr, spearmanr
        
        print("Analysis for SHAP vs Encoder Attention comparison for individual abnormal samples")
        print(f"Include self attention: {include_self_attention}")
        print(f"Plot group average: {plot_group_average}")
        self.model.eval()
        
        X_train, y_train = [], []
        for X, y in self.train_loader:
            X_train.append(X.cpu().numpy())
            y_train.append(y.cpu().numpy())
        
        X_test, y_test = [], []
        for X, y in self.test_loader:
            X_test.append(X.cpu().numpy())
            y_test.append(y.cpu().numpy())
        
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        # Combine train and test for supervised learning
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        print(f"Total data shape: {X_all.shape}, Num normal={np.sum(y_all==0)}, Num Abnormal={np.sum(y_all!=0)}")
        print("Training XGBoost")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42
        )
        xgb_model.fit(X_all, y_all)
        print("Training Done")
        
        # get shap values for all samples
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_all)  # Shape: (N_samples, N_features)
        
        # get encoder attn map
        print("Calculating encoder attention map")
        all_encoder_attention = []
        sample_indices = []
        current_idx = 0
        
        def compute_attention_with_self(attn_enc, attn_self_list):
            """
            Compute attention including self-attention layers
            attn_enc: (B, H, N, F)
            attn_self_list: List of (B, H, N, N)
            Returns: (B, F) - attention per feature
            """
            B, H, N, F = attn_enc.shape
            
            # Start with encoder attention: (B, H, N, F)
            # Average over heads first: (B, N, F)
            enc_mean = attn_enc.mean(dim=1)
            
            if attn_self_list and len(attn_self_list) > 0:
                # Compute cumulative self-attention: enc @ self_1 @ self_2 @ ... @ self_L
                # Each self attention: (B, H, N, N)
                # Average over heads: (B, N, N)
                cumulative_self = torch.eye(N, device=attn_enc.device).unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
                
                for self_attn in attn_self_list:
                    self_attn_mean = self_attn.mean(dim=1)  # (B, N, N)
                    # Add residual connection (identity matrix) for self-attention
                    self_attn_with_residual = self_attn_mean + torch.eye(N, device=attn_enc.device).unsqueeze(0).expand(B, -1, -1)
                    # Multiply: cumulative_self @ self_attn_with_residual
                    cumulative_self = torch.bmm(cumulative_self, self_attn_with_residual)  # (B, N, N)
                
                # Apply cumulative self-attention to encoder attention
                # cumulative_self: (B, N, N), enc_mean: (B, N, F)
                final_attn = torch.bmm(cumulative_self, enc_mean)  # (B, N, F)
            else:
                final_attn = enc_mean  # (B, N, F)
            
            # Average over latent dimensions to get per-feature attention: (B, F)
            return final_attn.mean(dim=1).detach().cpu().numpy()
        
        # trainset
        for X, y in self.train_loader:
            X = X.to(self.device)
            loss, attn_enc, attn_self_list, attn_dec = self.model(X, return_attn_weight=True)
            
            if include_self_attention:
                enc_attn_per_feature = compute_attention_with_self(attn_enc, attn_self_list)
            else:
                # attn_enc shape: (B, H, N, F) -> (B, F)
                enc_attn_per_feature = attn_enc.mean(dim=(1, 2)).detach().cpu().numpy()  # (B, F)
            
            all_encoder_attention.append(enc_attn_per_feature)
            
            batch_size = X.shape[0]
            sample_indices.extend(list(range(current_idx, current_idx + batch_size)))
            current_idx += batch_size
        
        # testset
        for X, y in self.test_loader:
            X = X.to(self.device)
            loss, attn_enc, attn_self_list, attn_dec = self.model(X, return_attn_weight=True)
            
            if include_self_attention:
                enc_attn_per_feature = compute_attention_with_self(attn_enc, attn_self_list)
            else:
                enc_attn_per_feature = attn_enc.mean(dim=(1, 2)).detach().cpu().numpy()
            
            all_encoder_attention.append(enc_attn_per_feature)
            
            batch_size = X.shape[0]
            sample_indices.extend(list(range(current_idx, current_idx + batch_size)))
            current_idx += batch_size
        
        all_encoder_attention = np.concatenate(all_encoder_attention, axis=0) # (N_total, F)
        
        print(f"Encoder attention shape: {all_encoder_attention.shape}")
        print(f"SHAP values shape: {shap_values.shape}")
        
        abnormal_mask = y_all != 0
        abnormal_indices = np.where(abnormal_mask)[0]
        
        if len(abnormal_indices) == 0:
            print("No abnormal samples found")
            return None
        
        print(f"Found {len(abnormal_indices)} abnormal samples")
        
        # Consider abnormal samples
        shap_abnormal = np.abs(shap_values[abnormal_mask])  # absolute SHAP
        attention_abnormal = np.abs(all_encoder_attention[abnormal_mask])  # Use absolute attention, not required.
        X_abnormal = X_all[abnormal_mask]
        
        # Option 1: Plot group averages (normal vs abnormal)
        if plot_group_average:
            print("Creating group average plots...")
            normal_mask = y_all == 0
            normal_indices = np.where(normal_mask)[0]
            
            if len(normal_indices) > 0:
                # Normal group averages
                shap_normal_mean = np.abs(shap_values[normal_mask]).mean(axis=0)
                attention_normal_mean = np.abs(all_encoder_attention[normal_mask]).mean(axis=0)
                
                # Abnormal group averages
                shap_abnormal_mean = shap_abnormal.mean(axis=0)
                attention_abnormal_mean = attention_abnormal.mean(axis=0)
                
                # Normalize for visualization
                shap_normal_norm = shap_normal_mean / (np.max(shap_normal_mean) + 1e-8)
                attention_normal_norm = attention_normal_mean / (np.max(attention_normal_mean) + 1e-8)
                shap_abnormal_norm = shap_abnormal_mean / (np.max(shap_abnormal_mean) + 1e-8)
                attention_abnormal_norm = attention_abnormal_mean / (np.max(attention_abnormal_mean) + 1e-8)
                
                feature_indices = np.arange(len(shap_normal_mean))
                
                # Normal group plot
                fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)
                axes[0].bar(feature_indices, shap_normal_norm, alpha=0.7, color='red', width=0.8)
                axes[0].set_title('SHAP Values', fontsize=14)
                axes[0].set_xlabel('Feature Index', fontsize=14)
                axes[0].set_ylabel('Normalized SHAP value', fontsize=14)
                axes[0].grid(True, alpha=0.3)
                
                axes[1].bar(feature_indices, attention_normal_norm, alpha=0.7, color='blue', width=0.8)
                axes[1].set_title('Encoder Attention', fontsize=14)
                axes[1].set_xlabel('Feature Index', fontsize=14)
                axes[1].set_ylabel('Normalized Attention Weight', fontsize=14)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                base_path = self.train_config['base_path']
                attention_type = "with_self" if include_self_attention else "enc_only"
                normal_plot_path = os.path.join(base_path, f'shap_vs_attention_normal_average_{attention_type}.png')
                plt.savefig(normal_plot_path, bbox_inches='tight', dpi=200)
                plt.close()
                print(f"Saved normal group average plot to {normal_plot_path}")
                
                # Abnormal group plot
                fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)
                axes[0].bar(feature_indices, shap_abnormal_norm, alpha=0.7, color='red', width=0.8)
                axes[0].set_title('SHAP Values', fontsize=14)
                axes[0].set_xlabel('Feature Index', fontsize=14)
                axes[0].set_ylabel('Normalized SHAP value', fontsize=14)
                axes[0].grid(True, alpha=0.3)
                
                axes[1].bar(feature_indices, attention_abnormal_norm, alpha=0.7, color='blue', width=0.8)
                axes[1].set_title('Encoder Attention', fontsize=14)
                axes[1].set_xlabel('Feature Index', fontsize=14)
                axes[1].set_ylabel('Normalized Attention Weight', fontsize=14)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                abnormal_plot_path = os.path.join(base_path, f'shap_vs_attention_abnormal_average_{attention_type}.png')
                plt.savefig(abnormal_plot_path, bbox_inches='tight', dpi=200)
                plt.close()
                print(f"Saved abnormal group average plot to {abnormal_plot_path}")
                
                # Calculate correlations for group averages
                normal_pearson_r, _ = pearsonr(shap_normal_mean, attention_normal_mean)
                normal_spearman_r, _ = spearmanr(shap_normal_mean, attention_normal_mean)
                abnormal_pearson_r, _ = pearsonr(shap_abnormal_mean, attention_abnormal_mean)
                abnormal_spearman_r, _ = spearmanr(shap_abnormal_mean, attention_abnormal_mean)
                
                print(f"Normal group correlations - Pearson: {normal_pearson_r:.4f}, Spearman: {normal_spearman_r:.4f}")
                print(f"Abnormal group correlations - Pearson: {abnormal_pearson_r:.4f}, Spearman: {abnormal_spearman_r:.4f}")
            else:
                print("No normal samples found for group average plotting")

        # calculate pearson and spearman for each sample between SHAP and Attn
        correlations = []
        for i in range(len(abnormal_indices)):
            shap_sample = shap_abnormal[i]
            attention_sample = attention_abnormal[i]

            pearson_r, pearson_p = pearsonr(shap_sample, attention_sample)
            spearman_r, spearman_p = spearmanr(shap_sample, attention_sample)
            
            correlations.append({
                'abnormal_idx': i,
                'global_idx': abnormal_indices[i],
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'max_correlation': max(abs(pearson_r), abs(spearman_r))
            })

        # sort and save nice ones        
        correlations.sort(key=lambda x: x['max_correlation'], reverse=True)
        similar_samples = [c for c in correlations if c['max_correlation'] >= similarity_threshold]
        similar_samples += [c for c in correlations if c['max_correlation'] <= -similarity_threshold]
        print(f"Found {len(similar_samples)} samples with correlation >= {similarity_threshold} or correlation <= {-similarity_threshold}")
        
        if len(similar_samples) == 0:
            print(f"No samples found with correlation >= {similarity_threshold}")
            # save top-k but not high correlation between SHAP and Attn
            similar_samples = correlations[:min(top_k_samples, len(correlations))]
            similar_samples += correlations[-min(top_k_samples, len(correlations)):]
            print(f"Saving top {len(similar_samples)} samples instead")
        
        # Save plots for top samples
        base_path = self.train_config['base_path']
        saved_plots = []
        
        samples_to_plot = similar_samples[:min(top_k_samples, len(similar_samples))] # top-k
        samples_to_plot += similar_samples[-min(top_k_samples, len(similar_samples)):] # bottom-k
        
        for sample_info in samples_to_plot:
            abnormal_idx = sample_info['abnormal_idx']
            global_idx = sample_info['global_idx']
            pearson_r = sample_info['pearson_r']
            spearman_r = sample_info['spearman_r']
            
            # Get SHAP and Attn for current sample and normalize 
            shap_sample = shap_abnormal[abnormal_idx]
            attention_sample = attention_abnormal[abnormal_idx]
            
            shap_norm = shap_sample / (np.max(shap_sample) + 1e-8)
            attention_norm = attention_sample / (np.max(attention_sample) + 1e-8)
            
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)
            feature_indices = np.arange(len(shap_sample))
            
            # SHAP values
            axes[0].bar(feature_indices, shap_norm, alpha=0.7, color='red', width=0.8)
            axes[0].set_title(f'SHAP Values', fontsize=14)
            axes[0].set_xlabel('Feature Index', fontsize=14)
            axes[0].set_ylabel('Normalized SHAP value', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            # Encoder attention
            axes[1].bar(feature_indices, attention_norm, alpha=0.7, color='blue', width=0.8)
            axes[1].set_title(f'Encoder Attention', fontsize=14)
            axes[1].set_xlabel('Feature Index', fontsize=14)
            axes[1].set_ylabel('Normalized Attention Weight', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            attention_type = "with_self" if include_self_attention else "enc_only"
            plot_path = os.path.join(base_path, f'shap_vs_attention_sample_{global_idx}_corr_{sample_info["max_correlation"]:.3f}_{attention_type}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=200)
            plt.close()
            
            saved_plots.append({
                'sample_idx': global_idx,
                'abnormal_idx': abnormal_idx,
                'plot_path': plot_path,
                'correlations': sample_info
            })
            
            print(f"Saved plot for sample {global_idx} (correlation: {sample_info['max_correlation']:.3f}) to {plot_path}")
        
        # Create summary statistics
        all_correlations = [c['max_correlation'] for c in correlations]
        
        summary_stats = {
            'total_abnormal_samples': len(abnormal_indices),
            'samples_above_threshold': len([c for c in correlations if c['max_correlation'] >= similarity_threshold]),
            'mean_correlation': np.mean(all_correlations),
            'median_correlation': np.median(all_correlations),
            'max_correlation': np.max(all_correlations),
            'min_correlation': np.min(all_correlations),
            'std_correlation': np.std(all_correlations)
        }
        
        print("\n" + "="*60)
        print("SHAP vs Encoder Attention Comparison Summary")
        print("="*60)
        print(f"Total abnormal samples: {summary_stats['total_abnormal_samples']}")
        print(f"Samples above threshold ({similarity_threshold}): {summary_stats['samples_above_threshold']}")
        print(f"Mean correlation: {summary_stats['mean_correlation']:.4f}")
        print(f"Median correlation: {summary_stats['median_correlation']:.4f}")
        print(f"Max correlation: {summary_stats['max_correlation']:.4f}")
        print(f"Min correlation: {summary_stats['min_correlation']:.4f}")
        print(f"Std correlation: {summary_stats['std_correlation']:.4f}")
        print(f"Saved {len(saved_plots)} plots")
        
        return {
            'correlations': correlations,
            'saved_plots': saved_plots,
            'summary_stats': summary_stats,
            'similarity_threshold': similarity_threshold,
            'shap_values_abnormal': shap_abnormal,
            'attention_values_abnormal': attention_abnormal
        }