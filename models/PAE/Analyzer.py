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
from utils import aucPerformance, F1Performance
from models.PAE.Trainer import Trainer
from models.PAE.Supervised import PAESupervised
import math

def nearest_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict, analysis_config: dict):
        super().__init__(model_config, train_config)
        self.cum_memory_weight = True
        self.model_config = model_config
        self.train_config = train_config
        self.analysis_config = analysis_config
        self.model_supervsied = PAESupervised(
            **model_config
        ).to(self.device)

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


    def training_supervised(self):
        print(self.model_config)
        print(self.train_config)
        parameter_path = os.path.join(self.train_config['base_path'], 'model_supervised.pt')
        if os.path.exists(parameter_path):
            print(f"model.pt already exists at {parameter_path}. Skip training and load parameters.")
            
            self.model_supervsied.load_state_dict(torch.load(parameter_path))  # 
            self.model_supervsied.eval()
            return

        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split

        optimizer = optim.Adam(self.model_supervsied.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model_supervsied.train()
        print("Training Start.")

        for epoch in range(50):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                # note that current dataset has lable in input_features
                # thus, we extract it and use as label.
                x_input = x_input[:, :-1].to(self.device)
                y = x_input[:, -1].to(self.device).float().view(-1)
                y_pred = self.model_supervsied(x_input)
                y_pred = y_pred.view(-1)     
                loss = F.mse_loss(y_pred, y, reduction='mean')

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            self.logger.info(info.format(epoch,loss.cpu()))
        print("Supervised Training complete.")

        print("Saving")
        torch.save(self.model_supervsied.state_dict(), parameter_path)

    @torch.no_grad()
    def plot_attn_and_corr(self):
        # Extract attention weights (H, F, F)
        attn_weights = self.get_attn_weights(use_self_attn=True)  # Shape: (H, F, F)
        attn_weights_no_self = self.get_attn_weights(use_self_attn=False)  # Shape: (H, F, F)
        H, F, F = attn_weights.shape

        # Create a combined plot (1 row x (1 + 2H) columns)
        fig_combined, axes_combined = plt.subplots(1, 2*H + 1, figsize=(6 * (2*H + 1), 6), dpi=200)
        if H == 1:
            axes_combined = [axes_combined]  # Handle the case where there is only one head

        # Plot the correlation matrix first
        X_all = []
        for X, _ in self.train_loader:
            X_all.append(X[:, :-1])  # Excluding the label
        X_all = torch.cat(X_all, dim=0).cpu().numpy()

        # Compute the correlation matrix
        corr_matrix = np.corrcoef(X_all.T)

        ax_corr = axes_combined[0]
        im = ax_corr.imshow(corr_matrix, cmap='coolwarm', aspect='equal')
        ax_corr.set_title('Feature Correlation Matrix')
        ax_corr.set_xlabel('Feature Index')
        ax_corr.set_ylabel('Feature Index')
        fig_combined.colorbar(im, ax=ax_corr, shrink=0.8, pad=0.02)

        # Plot attention maps for each head with both self-attention and non-self-attention
        for h in range(H):
            ax_self = axes_combined[2 * h + 1]
            ax_no_self = axes_combined[2 * h + 2]

            # Attention with self
            attn_map_self = attn_weights[h].cpu().numpy()
            im_self = ax_self.imshow(attn_map_self, aspect='equal', cmap='viridis')
            ax_self.set_title(f'Head {h+1} (Self Attention)')
            ax_self.set_xlabel('Input Feature Index')
            ax_self.set_ylabel('Output Feature Index')
            fig_combined.colorbar(im_self, ax=ax_self, shrink=0.8, pad=0.02)

            # Attention without self
            attn_map_no_self = attn_weights_no_self[h].cpu().numpy()
            im_no_self = ax_no_self.imshow(attn_map_no_self, aspect='equal', cmap='viridis')
            ax_no_self.set_title(f'Head {h+1} (No Self Attention)')
            ax_no_self.set_xlabel('Input Feature Index')
            ax_no_self.set_ylabel('Output Feature Index')
            fig_combined.colorbar(im_no_self, ax=ax_no_self, shrink=0.8, pad=0.02)

        # Adjust layout and save the figure
        fig_combined.tight_layout()
        
        # Save the figure in PNG and PDF formats
        base_path = self.train_config['base_path']
        os.makedirs(base_path, exist_ok=True)
        
        # Save combined plot
        combined_plot_path_png = os.path.join(base_path, 'attn_and_corr_combined.png')
        combined_plot_path_pdf = os.path.join(base_path, 'attn_and_corr_combined.pdf')
        fig_combined.savefig(combined_plot_path_png)
        fig_combined.savefig(combined_plot_path_pdf)
        print(f"Attention maps and correlation matrix saved to {combined_plot_path_png}.")

    @torch.no_grad()
    def compare_regresssion_with_attn(
        self,
        use_sup_attn: bool = False,
        lambda_attn: float = 1.0,
    ):  
        if use_sup_attn:
            attn_with_self = self.get_sup_attn_weights(use_self_attn=True, lambda_attn=lambda_attn)
            attn_no_self = self.get_sup_attn_weights(use_self_attn=False)
        
            attn_label_with_self = attn_with_self.cpu().numpy()
            attn_label_no_self = attn_no_self.cpu().numpy()
            # print(attn_label_no_self.shape)

            attn_label_with_self = attn_with_self[:, -1, :].cpu().numpy()
            attn_label_no_self = attn_no_self[:, -1, :].cpu().numpy()

        else:    
            attn_with_self = self.get_attn_weights(use_self_attn=True, lambda_attn=lambda_attn)
            attn_no_self = self.get_attn_weights(use_self_attn=False)

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
        if use_sup_attn:
            path = os.path.join(base_path, f'sup_attn_regression_comparison_lambda{lambda_attn}.csv')
        else:
            path = os.path.join(base_path, f'attn_regression_comparison_{lambda_attn}.csv')
        
        df.to_csv(path, index=True)
        
        print('Attention vs. Regression Comparison\n', df)
        return df


    @torch.no_grad()
    def get_sup_attn_weights(
        self,
        use_self_attn: bool = True,
        lambda_attn: float = 1.0,
    ):
        
        self.model_supervsied.eval()
        running = None   # (H, 1, F) accumulator (sum over batch)
        total_samples = 0

        for (X, y) in self.train_loader:
            X_input = X[:, :-1].to(self.device)
            y_pred, attn_enc, attn_self_list, attn_dec = \
                self.model_supervsied(X_input, return_weight=True)
            
            B, H, N, F = attn_enc.shape # N should be 1

            if use_self_attn and attn_self_list:
                I = torch.eye(N, device=self.device).expand(B, H, N, N)  # (B,H,N,N)
                W_self_total = attn_self_list[0] + I
                for W in attn_self_list[1:]:
                    W_with_res = lambda_attn * W + I
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
                running = attn_feat_feat.sum(dim=0)  # (H, 1, F)
            else:
                running += attn_feat_feat.sum(dim=0)

            total_samples += B

        result = running / total_samples  # (H, 1, F)
        return result

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

        for (X, y) in self.train_loader:
            X_input = X.to(self.device)
            loss, attn_enc, attn_self_list, attn_dec = \
                self.model(X_input, return_weight=True)

            B, H, N, F = attn_enc.shape

            
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
        return result


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

        x = torch.cat([x_norm, x_abn], dim=0).to(self.device)
        y = torch.cat([y_norm, y_abn], dim=0).to(self.device)

        batch_size, num_features = x.shape
        H = int(np.sqrt(num_features))
        assert H * H == num_features, f"{num_features} is not squared number."

        _, x, recon, encoder_attn, self_attns, decoder_attn = self.model(x, return_weight=True)

        base_path = self.model_config['base_path']
        os.makedirs(base_path, exist_ok=True)
        
        x_np = x.detach().float().cpu().numpy()
        recon_np = recon.detach().float().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        encoder_attn_np = encoder_attn.detach().float().cpu().numpy()
        self_attns_np = [sa.detach().float().cpu().numpy() for sa in self_attns]
        decoder_attn_np = decoder_attn.detach().float().cpu().numpy()

        def _norm_with_ref(arr, ref_min, ref_max):
            if ref_max > ref_min:
                return (arr - ref_min) / (ref_max - ref_min)
            return np.zeros_like(arr)

        saved_paths = []
        for i in range(batch_size):
            label = int(y_np[i])
            
            # --- [1] Reconstruction Plot (변경 없음) ---
            orig_img  = x_np[i].reshape(H, H)
            recon_img = recon_np[i].reshape(H, H)
            ref_min, ref_max = orig_img.min(), orig_img.max()
            orig_plot  = _norm_with_ref(orig_img,  ref_min, ref_max)
            recon_plot = _norm_with_ref(recon_img, ref_min, ref_max)

            plt.figure(figsize=(9, 3), dpi=200)
            plt.subplot(1, 2, 1)
            plt.imshow(orig_plot, cmap='gray', vmin=0, vmax=1)
            plt.title(f'Original (label={label})')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(recon_plot, cmap='gray', vmin=0, vmax=1)
            plt.title('Reconstruction')
            plt.axis('off')

            plt.tight_layout()
            out_path = os.path.join(base_path, f'reconstruction_pair_{i}_label{label}.png')
            plt.savefig(out_path)
            plt.close()
            saved_paths.append(out_path)

            # --- [2] Encoder Cross-Attention per Head (수정된 부분) ---
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200, constrained_layout=True)
            fig.suptitle(f'Sample {i} (label={label})\nEncoder Cross-Attention per Head', fontsize=16)
            
            vmin, vmax = encoder_attn_np[i].min(), encoder_attn_np[i].max()
            
            for h in range(4):
                ax = axes[h // 2, h % 2]
                enc_attn_map_head = encoder_attn_np[i][h]
                im = ax.imshow(enc_attn_map_head, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f'Head {h+1}')
                ax.set_xlabel('Input Features')
                ax.set_ylabel('Latent Queries')

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
            out_path_enc = os.path.join(base_path, f'attn_encoder_{i}_label{label}.png')
            fig.savefig(out_path_enc)
            plt.close(fig)
            saved_paths.append(out_path_enc)

            # --- [3] Self-Attention Blocks per Head (수정된 부분) ---
            num_blocks = len(self_attns_np)
            for j in range(num_blocks):
                fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200, constrained_layout=True)
                fig.suptitle(f'Sample {i} (label={label})\nSelf-Attention Block {j+1} per Head', fontsize=16)

                block_attns = self_attns_np[j][i]
                vmin, vmax = block_attns.min(), block_attns.max()

                for h in range(4):
                    ax = axes[h // 2, h % 2]
                    self_attn_map_head = block_attns[h]
                    im = ax.imshow(self_attn_map_head, aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Head {h+1}')
                    ax.set_xlabel('Latent Keys')
                    ax.set_ylabel('Latent Queries')

                fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
                out_path_self = os.path.join(base_path, f'attn_self_block{j+1}_{i}_label{label}.png')
                fig.savefig(out_path_self)
                plt.close(fig)
                saved_paths.append(out_path_self)

            # --- [4] Decoder Cross-Attention per Head (수정된 부분) ---
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200, constrained_layout=True)
            fig.suptitle(f'Sample {i} (label={label})\nDecoder Cross-Attention per Head', fontsize=16)

            vmin, vmax = decoder_attn_np[i].min(), decoder_attn_np[i].max()

            for h in range(4):
                ax = axes[h // 2, h % 2]
                dec_attn_map_head = decoder_attn_np[i][h]
                im = ax.imshow(dec_attn_map_head, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f'Head {h+1}')
                ax.set_xlabel('Latent Keys')
                ax.set_ylabel('Output Queries')
            
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
            out_path_dec = os.path.join(base_path, f'attn_decoder_{i}_label{label}.png')
            fig.savefig(out_path_dec)
            plt.close(fig)
            saved_paths.append(out_path_dec)

 
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

        base_path = self.model_config['base_path']
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
        plt.title(f"Anomaly Score on {self.model_config['dataset_name'].upper()}{title_suffix}")
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

        fig.suptitle(f"Anomaly Score Distributions • {self.model_config['dataset_name'].upper()}{title_suffix}",
                    y=1.02, fontsize=11)
        fig.tight_layout()
        out_grid = os.path.join(base_path, "hist_anomaly_score_1x3.png")
        fig.savefig(out_grid, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved histogram to {out_overlay}, {out_grid}")


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

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            loss = self.model(x_input)
            loss = loss.data.cpu()
            score.append(loss)
            test_label.append(y_label)
        score = torch.cat(score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        return rauc, ap, f1