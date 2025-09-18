import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from models.MemPAE.Trainer import Trainer

class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict, analysis_config: dict):
        super().__init__(model_config, train_config)

        self.plot_attn = analysis_config['plot_attn'] 
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

    def plot_attn(self,):
        
        # get attention per head
        attn_feature_feature = self.get_attn_weights(use_self_attn=True) # (H, F, F)
        
        
        attn_feature_feature = self.get_attn_weights(use_self_attn=False) # (H, F, F)

        # calculate corr matrix in trainset.
        # 


        return


    @torch.no_grad()
    def plot_combined_tsne(self, figsize=(10, 8)):
        """
        normal_input, abnormal_input, abnormal_recon_mem 세 가지 데이터를
        하나의 t-SNE 공간에 시각화하여 분포를 비교합니다.
        """
        self.model.eval()
        print("데이터 수집 및 전처리 중...")
        recons = self._accumulate_recon()

        # 1. 시각화에 사용할 세 가지 데이터셋을 선택합니다.
        normal_input = recons['normal_input']
        abnormal_input = recons['abnormal_input']
        abnormal_recon_mem = recons['abnormal_recon_mem']
        
        datasets = [normal_input, abnormal_input, abnormal_recon_mem]
        
        if any(d.shape[0] == 0 for d in datasets):
            print("경고: 데이터셋 중 일부가 비어있어 시각화를 건너뜁니다.")
            return

        # 2. 모든 데이터를 하나로 합치고, 각 그룹을 구분할 레이블을 생성합니다.
        # 레이블: 0=Normal Input, 1=Abnormal Input, 2=Abnormal Recon (Mem)
        labels = []
        for i, d in enumerate(datasets):
            labels.extend([i] * d.shape[0])
        labels = np.array(labels)
        
        combined_data = np.vstack(datasets)

        # 3. 합쳐진 전체 데이터에 대해 t-SNE를 실행합니다.
        print("t-SNE 변환을 시작합니다...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(scaled_data)
        print("t-SNE 변환 완료.")

        # --- 4. 통합 플롯 그리기 ---
        fig, ax = plt.subplots(figsize=figsize)
        
        # 각 그룹에 대한 이름, 색상, 마커 정의
        plot_info = {
            0: {'label': 'Normal Input', 'color': 'royalblue', 'marker': 'o', 'alpha': 0.5, 's': 50},
            1: {'label': 'Abnormal Input', 'color': 'darkorange', 'marker': 'X', 'alpha': 0.8, 's': 70},
            2: {'label': 'Abnormal Recon (Mem)', 'color': 'green', 'marker': 's', 'alpha': 0.8, 's': 70}
        }

        # 각 그룹별로 점을 찍습니다.
        for label_id, info in plot_info.items():
            mask = (labels == label_id)
            ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                    label=info['label'], c=info['color'], 
                    marker=info['marker'], alpha=info['alpha'], s=info['s'])

        ax.set_title('Combined t-SNE Visualization', fontsize=16)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, 't-sne_combined_visualization.png')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"통합 t-SNE 시각화가 '{out_path}'에 저장되었습니다.")

    # -------------------------------
    # 1) 수집: model(x, return_pred_all=True) -> (x, x_hat_origin, x_hat_memory) 가정
    # -------------------------------
    @torch.no_grad()
    def _collect_recon(self, loader):
        x_list, x_hat_origin_list, x_hat_memory_list, labels = [], [], [], []
        self.model.eval()

        for batch in loader:
            # (x, y) 또는 (x,) 형태 모두 대비
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x, y = batch, None

            x = x.to(self.device)

            # model forward
            x_out, x_hat_origin, x_hat_memory = self.model(x, return_pred_all=True)

            # numpy로 변환
            x_np = x_out.detach().cpu().numpy()
            x_hat_origin_np = x_hat_origin.detach().cpu().numpy()
            x_hat_memory_np = x_hat_memory.detach().cpu().numpy()

            # label 처리
            if y is None:
                y_arr = np.zeros(x_np.shape[0], dtype=np.int64)
            else:
                y_arr = y.view(-1).detach().cpu().numpy()

            x_list.append(x_np)
            x_hat_origin_list.append(x_hat_origin_np)
            x_hat_memory_list.append(x_hat_memory_np)
            labels.append(y_arr)

        x_list = np.concatenate(x_list, axis=0)
        x_hat_origin_list = np.concatenate(x_hat_origin_list, axis=0)
        x_hat_memory_list = np.concatenate(x_hat_memory_list, axis=0)
        labels = np.concatenate(labels, axis=0)

        return x_list, x_hat_origin_list, x_hat_memory_list, labels


    # -------------------------------
    # 2) 누적/정리: base는 항상 normal_input (라벨==0)
    #    abnormal은 라벨==1 로 가정(다중 클래스면 !=0 으로 바꿔도 됨)
    # -------------------------------
    @torch.no_grad()
    def _accumulate_recon(self):
        self.model.eval()
        test_input, test_recon, test_recon_mem, test_labels = self._collect_recon(self.test_loader)

        # 베이스: normal
        normal_mask = (test_labels == 0)
        normal_input     = test_input[normal_mask]
        normal_recon     = test_recon[normal_mask]
        normal_recon_mem = test_recon_mem[normal_mask]

        # 오버레이: abnormal (여기서는 1로 명시)
        abnormal_mask = (test_labels == 1)
        abnormal_input     = test_input[abnormal_mask]
        abnormal_recon     = test_recon[abnormal_mask]
        abnormal_recon_mem = test_recon_mem[abnormal_mask]

        output = {
            'normal_input': normal_input,
            'normal_recon': normal_recon,
            'normal_recon_mem': normal_recon_mem,
            'abnormal_input': abnormal_input,
            'abnormal_recon': abnormal_recon,
            'abnormal_recon_mem': abnormal_recon_mem,
        }
        return output


    @torch.no_grad()
    def plot_tsne_reconstruction(self, figsize=(16, 5), s_base=12, s_overlay=24, alpha_base=0.35, alpha_overlay=0.9):
        """
        패널별로 항상 'normal_input'을 베이스로 깔고,
        그 위에 abnormal_* (input / recon / recon_mem)를 오버레이로 시각화.
        """
        def _fit_tsne_on_base_overlay(base_X, overlay_X, perplexity=30, random_state=42, n_iter=1000):
                """
                base_X로만 StandardScaler를 fit하고, base/overlay를 동일 스케일로 변환.
                변환된 두 집합을 concat하여 t-SNE를 fit → 같은 2D 공간으로 임베딩.
                """
                base_X = np.asarray(base_X, dtype=np.float32)
                overlay_X = np.asarray(overlay_X, dtype=np.float32)
                n_base = base_X.shape[0]
                n_overlay = overlay_X.shape[0]
                n_total = n_base + n_overlay

                if n_total < 5 or n_overlay == 0:
                    # 샘플 수가 너무 적거나 overlay가 없으면 빈 결과 반환
                    return None, None, False

                # t-SNE 안정화를 위해 베이스 기준 스케일링
                scaler = StandardScaler().fit(base_X)
                base_Z = scaler.transform(base_X)
                overlay_Z = scaler.transform(overlay_X)

                X = np.concatenate([base_Z, overlay_Z], axis=0)

                # perplexity는 (n_samples - 1) 보다 작아야 하고, 최소 5 정도 권장
                safe_perp = max(5, min(perplexity, (n_total - 1) // 3 if n_total > 10 else 5))

                tsne = TSNE(
                    n_components=2,
                    perplexity=safe_perp,
                    learning_rate='auto',
                    init='pca',
                    n_iter=n_iter,
                    random_state=random_state,
                    verbose=0,
                    metric='euclidean',
                    n_jobs=None  # 최신 sklearn은 n_jobs 인자 없음(버전에 맞춰 조정)
                )
                Y = tsne.fit_transform(X)
                Y_base = Y[:n_base]
                Y_overlay = Y[n_base:]
                return Y_base, Y_overlay, True

        self.model.eval()
        recons = self._accumulate_recon()

        normal_input     = recons['normal_input']
        abnormal_input   = recons['abnormal_input']
        abnormal_recon   = recons['abnormal_recon']
        abnormal_recon_mem = recons['abnormal_recon_mem']

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        panels = [
            ("Input: normal (base) + abnormal (overlay)",      abnormal_input),
            ("Recon: normal (base) + abnormal Dec(z) (overlay)",   abnormal_recon),
            ("Recon (Mem): normal (base) + abnormal Dec(ẑ) (overlay)", abnormal_recon_mem),
        ]

        for ax, (title, overlay_X) in zip(axes, panels):
            Y_base, Y_overlay, ok = _fit_tsne_on_base_overlay(normal_input, overlay_X)
            ax.set_title(title)

            if not ok:
                ax.text(0.5, 0.5, "Not enough samples to plot", ha='center', va='center')
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # base=normal을 먼저(뒤) 그리면 overlay가 가려질 수 있어 overlay를 나중에 그림
            ax.scatter(Y_base[:, 0], Y_base[:, 1], s=s_base, alpha=alpha_base, label="normal (base)")
            ax.scatter(Y_overlay[:, 0], Y_overlay[:, 1], s=s_overlay, alpha=alpha_overlay, label="abnormal (overlay)")

            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
            ax.legend(loc='best')

        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f't-sne_visualizaion.png')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


    @torch.no_grad()
    def compare_regresssion_with_attn(self):
        attn_with_self = self.get_attn_weights(use_self_attn=True)
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
        path = os.path.join(base_path, 'attn_regression_comparison.csv')
        df.to_csv(path, index=True)
        
        print('Attention vs. Regression Comparison\n', df)
        return df

    @torch.no_grad()
    def get_attn_weights(
        self,
        use_self_attn: bool = True,
    ):
        '''
        Get feature-feature attention weights.
        W_enc, W_self_1, ..., W_self_L, W_dec
        W_enc: (B, H, N, F)
        W_self_i: (B, H, N, N)
        W_dec: (B, H, F, N)

        return: (H, F, F)
        '''
        self.model.eval() # Set the model to evaluation mode
        running_attn_feature_feature = None
        total_samples = 0

        for (X, y) in self.train_loader:
            X_input = X.to(self.device)

            # Get attention weights from the model
            loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = \
                self.model(X_input, return_attn_weight=True)

            B, H, N, _ = attn_weight_enc.shape # Batch, Head, Node, Feature

            if use_self_attn and attn_weight_self_list:
                identity = torch.eye(N, device=self.device).expand(B, H, N, N)
                W_self_total = attn_weight_self_list[0] + identity

                for W in attn_weight_self_list[1:]:
                    W_with_residual = W + identity
                    W_self_total = torch.einsum("bhij,bhjk->bhik", W_with_residual, W_self_total)
            else:
                # for W_dec x W_enc
                W_self_total = torch.eye(N, device=self.device).expand(B, H, N, N)
            
            # Compose all weights: W_dec @ W_self @ W_enc
            attn_feature_feature = torch.einsum("bhfn,bhnn->bhfn",
                                                attn_weight_dec, W_self_total)
            attn_feature_feature = torch.einsum("bhfn,bhnk->bhfk",
                                                attn_feature_feature, attn_weight_enc)
            
            # Accumulate attention weights across all batches
            batch_size = attn_feature_feature.shape[0]
            if running_attn_feature_feature is None:
                running_attn_feature_feature = attn_feature_feature.sum(dim=0) # (H, F, F)
            else:
                running_attn_feature_feature += attn_feature_feature.sum(dim=0)
            
            total_samples += batch_size

        # Average the weights over all samples
        running_attn_feature_feature = running_attn_feature_feature / total_samples
        return running_attn_feature_feature # (H, F, F)

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