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
import umap.umap_ as umap
from scipy.stats import spearmanr
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
        file_name = "pos_encoding+token.png" if use_mask else pos_encoding.png
        out_path = os.path.join(base_path, file_name)

        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"pos_encoding saved into '{out_path}'.")
        
        return

    @torch.no_grad()
    def decode_each_memory(
        self,
    ):
        self.model.eval()
        # (num_memory, D) -> consider as (batch_size, hidden_dim)
        # decode each memory
        memory_latents = self.model.memory.memories
        num_memories = memory_latents.shape[0]  
        memory_latents = memory_latents.unsqueeze(1)
        decoder_query = self.model.decoder_query.expand(num_memories, self.model.num_features, -1)
        decoder_query = decoder_query + self.model.pos_encoding

        output = self.model.decoder(decoder_query, memory_latents, memory_latents)
        x_hat = self.model.proj(output)  # (M, F) => need to visualize on TSNE

        return x_hat


    @torch.no_grad()
    def plot_tsne_original_with_memory(
        self,
    ):
        self.model.eval()
        
        # Get original data from test loader
        all_original_data = []
        all_labels = []
        
        for (X, y) in self.test_loader:
            all_original_data.append(X.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        
        original_data = np.concatenate(all_original_data, axis=0)  # (N, F)
        labels = np.concatenate(all_labels, axis=0)  # (N,)
        
        # Separate normal and abnormal
        normal_mask = (labels == 0)
        abnormal_mask = (labels == 1)
        
        normal_data = original_data[normal_mask]
        abnormal_data = original_data[abnormal_mask]
        
        # Get decoded memory vectors
        decoded_memory = self.decode_each_memory().detach().cpu().numpy()  # (M, F)
        
        # Prepare data for t-SNE
        datasets = []
        category_labels = []
        
        # Add normal data
        if normal_data.shape[0] > 0:
            datasets.append(normal_data)
            category_labels.extend([0] * normal_data.shape[0])
        
        # Add abnormal data  
        if abnormal_data.shape[0] > 0:
            datasets.append(abnormal_data)
            category_labels.extend([1] * abnormal_data.shape[0])
        
        # Add decoded memory
        datasets.append(decoded_memory)
        category_labels.extend([2] * decoded_memory.shape[0])
        
        # Combine all data
        combined_data = np.vstack(datasets)
        category_labels = np.array(category_labels)
        
        # Apply t-SNE
        n_samples = combined_data.shape[0]
        perplexity = min(30, max(5, n_samples // 3)) if n_samples > 10 else 5
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            n_iter=1000, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results = tsne.fit_transform(combined_data)
        
        # Create the plot
        plt.figure(figsize=(10, 8), dpi=200)
        
        # Define colors and markers for each category
        plot_config =  {
            0: {'label': 'Normal (Original)', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.6, 's': 100},
            1: {'label': 'Abnormal (Origina)', 'color': '#d62728', 'marker': '^', 'alpha': 0.6, 's': 100}, 
            2: {'label': 'Decoded Memory', 'color': '#2ca02c', 'marker': 's', 'alpha': 0.8, 's': 150}
        }
        # plot_config = {
        #     0: {'label': 'Normal (Original)', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.6, 's': 30},
        #     1: {'label': 'Abnormal (Original)', 'color': '#d62728', 'marker': '^', 'alpha': 0.6, 's': 30}, 
        #     2: {'label': 'Decoded Memory', 'color': '#ff7f0e', 'marker': 's', 'alpha': 0.8, 's': 60}
        # }
        
        # Plot each category
        for cat_id, config in plot_config.items():
            mask = (category_labels == cat_id)
            if mask.any():
                plt.scatter(
                    tsne_results[mask, 0], 
                    tsne_results[mask, 1],
                    label=config['label'],
                    c=config['color'],
                    marker=config['marker'],
                    alpha=config['alpha'],
                    s=config['s'],
                    edgecolors='white',
                    linewidth=0.5
                )
        
        plt.title(f't-SNE: Original Data vs Decoded Memory • {self.train_config["dataset_name"].upper()}', 
                fontsize=14, pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, 't-sne_original_with_memory.png')

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Original data with decoded memory t-SNE saved into '{out_path}'.")



    @torch.no_grad()
    def get_latent(
        self,
    ):
        self.model.eval()
        all_labels = []
        all_latents = []
        all_latents_hat = []
        for (X, y) in self.test_loader:
            X_input = X.to(self.device)
            _, latents, latents_hat = self.model(X_input, return_latents=True)
            latents = F.normalize(latents, dim=-1)
            latents_hat = F.normalize(latents_hat, dim=-1)
            B, N, D = latents.shape
            all_labels.append(y.cpu())
            all_latents.append(latents.cpu())
            all_latents_hat.append(latents_hat.cpu())

        latents = torch.cat(all_latents, dim=0) # (N_train, N, D) 
        latents_hat = torch.cat(all_latents_hat, dim=0) # (N_train, N, D)
        labels = torch.cat(all_labels, dim=0) # (N_train, )
        return latents, latents_hat, labels # 

    def plot_tsne_latent_vs_memory(
        self,
        use_latents_hat: bool = False,
        use_latents_avg: bool = False,
        use_both_latents: bool = False,  
        latent_idx: int = None,
    ):
        # get latents and memory vector
        latents, latents_hat, labels = self.get_latent()
        
        if latent_idx is not None:
            if use_both_latents:
                filename = f'both_latent{latent_idx}' 
            elif use_latents_hat:
                filename = f'latent_hat{latent_idx}'
            else:
                filename = f'latent{latent_idx}' 
        else:    
            if use_both_latents:
                filename = 'both_latents_avg' if use_latents_avg else 'both_latents'
            elif use_latents_hat:
                filename = 'latents_hat_avg' if use_latents_avg else 'latents_hat'
            else:
                filename = 'latents_avg' if use_latents_avg else 'latents'

        memory_latents = self.model.memory.memories  # (num_memory, D)
        memory_latents = F.normalize(memory_latents, dim=-1)
        memory_latents = memory_latents.detach().cpu().numpy()

        if latent_idx is not None:
            latents = latents[:, latent_idx, :]   
            latents_hat = latents_hat[:, latent_idx, :]   
            # latents_hat

        datasets = []
        category_labels = []
        
        if use_both_latents:
            # Both latents and latents_hat
            if not use_latents_avg:
                hidden_dim = latents.shape[-1]
                # Original latents
                normal_latents = latents[labels==0].view(-1, hidden_dim)
                abnormal_latents = latents[labels==1].view(-1, hidden_dim)
                # Reconstructed latents
                normal_latents_hat = latents_hat[labels==0].view(-1, hidden_dim)
                abnormal_latents_hat = latents_hat[labels==1].view(-1, hidden_dim)
            else:
                # Average over sequence dimension
                normal_latents = latents[labels==0].mean(axis=1)
                abnormal_latents = latents[labels==1].mean(axis=1)
                normal_latents_hat = latents_hat[labels==0].mean(axis=1)
                abnormal_latents_hat = latents_hat[labels==1].mean(axis=1)
            
            # Add all datasets
            datasets.append(normal_latents.numpy())
            category_labels.extend([0] * normal_latents.shape[0])  # normal latents
            
            datasets.append(normal_latents_hat.numpy())
            category_labels.extend([1] * normal_latents_hat.shape[0])  # normal latents_hat
            
            datasets.append(abnormal_latents.numpy())
            category_labels.extend([2] * abnormal_latents.shape[0])  # abnormal latents
            
            datasets.append(abnormal_latents_hat.numpy())
            category_labels.extend([3] * abnormal_latents_hat.shape[0])  # abnormal latents_hat
            
            datasets.append(memory_latents)
            category_labels.extend([4] * memory_latents.shape[0])  # memory vectors
            
        else:
            # Original behavior
            if use_latents_hat:
                latents = latents_hat

            if not use_latents_avg:
                hidden_dim = latents.shape[-1]
                normal_latents = latents[labels==0].view(-1, hidden_dim)
                abnormal_latents = latents[labels==1].view(-1, hidden_dim)
            else:
                normal_latents = latents[labels==0].mean(axis=1)
                abnormal_latents = latents[labels==1].mean(axis=1)
            
            datasets.append(normal_latents.numpy())
            category_labels.extend([0] * normal_latents.shape[0])
            
            datasets.append(abnormal_latents.numpy())
            category_labels.extend([1] * abnormal_latents.shape[0])
            
            datasets.append(memory_latents)
            category_labels.extend([2] * memory_latents.shape[0])
        
        # Combine all data
        combined_data = np.vstack(datasets)
        category_labels = np.array(category_labels)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
        # Apply t-SNE
        n_samples = scaled_data.shape[0]
        perplexity = min(30, max(5, n_samples // 3)) if n_samples > 10 else 5
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            n_iter=1000, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results = tsne.fit_transform(scaled_data)
        
        # Create the plot
        plt.figure(figsize=(10, 8), dpi=200)
        
        # Define colors and markers for each category
        if use_both_latents:
            plot_config = {
                0: {'label': 'Normal Latents', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.6, 's': 80},
                1: {'label': 'Normal Latents_hat', 'color': '#aec7e8', 'marker': 's', 'alpha': 0.6, 's': 80},
                2: {'label': 'Abnormal Latents', 'color': '#d62728', 'marker': '^', 'alpha': 0.6, 's': 80},
                3: {'label': 'Abnormal Latents_hat', 'color': '#ff9896', 'marker': 'v', 'alpha': 0.6, 's': 80},
                4: {'label': 'Memory Vectors', 'color': '#2ca02c', 'marker': 'D', 'alpha': 0.8, 's': 120}
            }
        else:
            plot_config = {
                0: {'label': 'Normal Latents', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.6, 's': 100},
                1: {'label': 'Abnormal Latents', 'color': '#d62728', 'marker': '^', 'alpha': 0.6, 's': 100}, 
                2: {'label': 'Memory Vectors', 'color': '#2ca02c', 'marker': 's', 'alpha': 0.8, 's': 150}
            }
        
        # Plot each category
        for cat_id, config in plot_config.items():
            mask = (category_labels == cat_id)
            if mask.any():
                plt.scatter(
                    tsne_results[mask, 0], 
                    tsne_results[mask, 1],
                    label=config['label'],
                    c=config['color'],
                    marker=config['marker'],
                    alpha=config['alpha'],
                    s=config['s'],
                    edgecolors='white',
                    linewidth=0.5
                )
        
        plt.title(f't-SNE: {filename} vs Memory Vectors • {self.train_config["dataset_name"].upper()}', 
                fontsize=14, pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f't-sne_{filename}_memory.png')

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"T-SNE saved into '{out_path}'.")

        
    def plot_tsne_memory_separate(
        self,
        use_latents_hat = False,
    ):
        
        # get latents and memory vector
        latents, latents_hat, labels = self.get_latent()
        if use_latents_hat:
            filename = 'latents_hat'
            latents = latents_hat
        else:
            filename = 'latents'
        
        hidden_dim = latents.shape[-1]
        normal_latents = latents[labels==0].view(-1, hidden_dim) # (num_normal x num_latent, D)
        abnormal_latents = latents[labels==1].view(-1, hidden_dim) # (num_abnormal x num_latent, D)
        memory_latents = self.model.memory.memories.detach().cpu().numpy() # (num_memory, D)

        # Create 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200)

        # Plot 1: Memory vs Normal
        datasets_normal = [normal_latents.numpy(), memory_latents]
        category_labels_normal = []
        category_labels_normal.extend([0] * normal_latents.shape[0])  # normal latents
        category_labels_normal.extend([1] * memory_latents.shape[0])  # memory vectors
        
        combined_data_normal = np.vstack(datasets_normal)
        category_labels_normal = np.array(category_labels_normal)
        
        # Standardize and apply t-SNE for normal
        scaler_normal = StandardScaler()
        scaled_data_normal = scaler_normal.fit_transform(combined_data_normal)
        
        n_samples_normal = scaled_data_normal.shape[0]
        perplexity_normal = min(30, max(5, n_samples_normal // 3)) if n_samples_normal > 10 else 5
        
        tsne_normal = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity_normal, 
            n_iter=1000, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results_normal = tsne_normal.fit_transform(scaled_data_normal)
        
        # Plot normal vs memory
        normal_mask = (category_labels_normal == 0)
        memory_mask = (category_labels_normal == 1)
        
        axes[0].scatter(
            tsne_results_normal[normal_mask, 0], 
            tsne_results_normal[normal_mask, 1],
            label='Normal Latents',
            c='#1f77b4',
            marker='o',
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
        
        axes[0].scatter(
            tsne_results_normal[memory_mask, 0], 
            tsne_results_normal[memory_mask, 1],
            label='Memory Vectors',
            c='#2ca02c',
            marker='s',
            alpha=0.8,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
        
        axes[0].set_title(f'Memory vs Normal {filename}', fontsize=12)
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        axes[0].legend(loc='best')
        axes[0].grid(True, linestyle='--', alpha=0.3)

        # Plot 2: Memory vs Abnormal
        datasets_abnormal = [abnormal_latents.numpy(), memory_latents]
        category_labels_abnormal = []
        category_labels_abnormal.extend([0] * abnormal_latents.shape[0])  # abnormal latents
        category_labels_abnormal.extend([1] * memory_latents.shape[0])  # memory vectors
        
        combined_data_abnormal = np.vstack(datasets_abnormal)
        category_labels_abnormal = np.array(category_labels_abnormal)
        
        # Standardize and apply t-SNE for abnormal
        scaler_abnormal = StandardScaler()
        scaled_data_abnormal = scaler_abnormal.fit_transform(combined_data_abnormal)
        
        n_samples_abnormal = scaled_data_abnormal.shape[0]
        perplexity_abnormal = min(30, max(5, n_samples_abnormal // 3)) if n_samples_abnormal > 10 else 5
        
        tsne_abnormal = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity_abnormal, 
            n_iter=1000, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results_abnormal = tsne_abnormal.fit_transform(scaled_data_abnormal)
        
        # Plot abnormal vs memory
        abnormal_mask = (category_labels_abnormal == 0)
        memory_mask_2 = (category_labels_abnormal == 1)
        
        axes[1].scatter(
            tsne_results_abnormal[abnormal_mask, 0], 
            tsne_results_abnormal[abnormal_mask, 1],
            label=f'Abnormal {filename}',
            c='#d62728',
            marker='^',
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
        
        axes[1].scatter(
            tsne_results_abnormal[memory_mask_2, 0], 
            tsne_results_abnormal[memory_mask_2, 1],
            label='Memory Vectors',
            c='#2ca02c',
            marker='s',
            alpha=0.8,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
        
        axes[1].set_title(f'Memory vs Abnormal {filename}', fontsize=12)
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].legend(loc='best')
        axes[1].grid(True, linestyle='--', alpha=0.3)

        # Overall title
        fig.suptitle(f't-SNE: Memory vs {filename} • {self.train_config["dataset_name"].upper()}', 
                    fontsize=14, y=1.02)
        
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f't-sne_memory_{filename}_separate.png')

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Separate t-SNE plots saved into '{out_path}'.")

    def plot_attn_normal_vs_abnormal(self,):
        attn_feature_feature, label = self.get_attn_weights(use_self_attn=True) # (H, F, F)
        

    def plot_attn(self,):
        
        # get attention per head
        attn_feature_feature, label = self.get_attn_weights(use_self_attn=True) # (H, F, F)
        
        
        attn_feature_feature, label = self.get_attn_weights(use_self_attn=False) # (H, F, F)

        # calculate corr matrix in trainset.
        # 


        return


    @torch.no_grad()
    def plot_combined_tsne(self, figsize=(10, 8)):
        self.model.eval()
        recons = self._accumulate_recon()

        normal_input = recons['normal_input']
        normal_recon = recons['normal_recon']
        abnormal_input = recons['abnormal_input']
        abnormal_recon_mem = recons['abnormal_recon_mem']
        
        datasets = [normal_input, normal_recon, abnormal_input, abnormal_recon_mem]
        
        if any(d.shape[0] == 0 for d in datasets):
            print("No data. Skip Vis.")
            return

        labels = []
        for i, d in enumerate(datasets):
            labels.extend([i] * d.shape[0])
        labels = np.array(labels)
        
        combined_data = np.vstack(datasets)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(scaled_data)

        fig, ax = plt.subplots(figsize=figsize)
        
        plot_info = {
            0: {'label': 'Normal Input', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.7, 's': 50}, # Blue (Matplotlib default blue)
            1: {'label': 'Normal Recon', 'color': '#aec7e8', 'marker': 's', 'alpha': 0.7, 's': 50}, # Light Blue
            2: {'label': 'Abnormal Input', 'color': '#d62728', 'marker': '^', 'alpha': 0.7, 's': 50}, # Red (Matplotlib default red)
            3: {'label': 'Abnormal Recon', 'color': '#ff9896', 'marker': 'D', 'alpha': 0.7, 's': 50}, # Light Red
        }
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
        print(f"T-SNE saved into '{out_path}'.")

    @torch.no_grad()
    def _collect_recon(self, loader):
        x_list, x_hat_origin_list, x_hat_memory_list, labels = [], [], [], []
        self.model.eval()

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x, y = batch, None

            x = x.to(self.device)
            x_out, x_hat_origin, x_hat_memory = self.model(x, return_pred_all=True)
            x_np = x_out.detach().cpu().numpy()
            x_hat_origin_np = x_hat_origin.detach().cpu().numpy()
            x_hat_memory_np = x_hat_memory.detach().cpu().numpy()

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


    @torch.no_grad()
    def _accumulate_recon(self):
        self.model.eval()
        test_input, test_recon, test_recon_mem, test_labels = self._collect_recon(self.test_loader)

        normal_mask = (test_labels == 0)
        normal_input     = test_input[normal_mask]
        normal_recon     = test_recon[normal_mask]
        normal_recon_mem = test_recon_mem[normal_mask]

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
        def _fit_tsne_on_base_overlay(base_X, overlay_X, perplexity=30, random_state=42, n_iter=1000):
                base_X = np.asarray(base_X, dtype=np.float32)
                overlay_X = np.asarray(overlay_X, dtype=np.float32)
                n_base = base_X.shape[0]
                n_overlay = overlay_X.shape[0]
                n_total = n_base + n_overlay

                if n_total < 5 or n_overlay == 0:
                    return None, None, False

                scaler = StandardScaler().fit(base_X)
                base_Z = scaler.transform(base_X)
                overlay_Z = scaler.transform(overlay_X)

                X = np.concatenate([base_Z, overlay_Z], axis=0)

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




    def plot_umap_latent_vs_memory(
        self,
        use_latents_hat: bool = False,
        use_latents_avg: bool = False,
        use_both_latents: bool = False,  #
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ):
        # get latents and memory vector
        latents, latents_hat, labels = self.get_latent()
        
        if use_both_latents:
            filename = 'both_latents_avg' if use_latents_avg else 'both_latents'
        elif use_latents_hat:
            filename = 'latents_hat_avg' if use_latents_avg else 'latents_hat'
        else:
            filename = 'latents_avg' if use_latents_avg else 'latents'

        memory_latents = self.model.memory.memories  # (num_memory, D)
        memory_latents = F.normalize(memory_latents, dim=-1)
        memory_latents = memory_latents.detach().cpu().numpy()
    
        datasets = []
        category_labels = []
        
        if use_both_latents:
            # Both latents and latents_hat
            if not use_latents_avg:
                hidden_dim = latents.shape[-1]
                # Original latents
                normal_latents = latents[labels==0].view(-1, hidden_dim)
                abnormal_latents = latents[labels==1].view(-1, hidden_dim)
                # Reconstructed latents
                normal_latents_hat = latents_hat[labels==0].view(-1, hidden_dim)
                abnormal_latents_hat = latents_hat[labels==1].view(-1, hidden_dim)
            else:
                # Average over sequence dimension
                normal_latents = latents[labels==0].mean(axis=1)
                abnormal_latents = latents[labels==1].mean(axis=1)
                normal_latents_hat = latents_hat[labels==0].mean(axis=1)
                abnormal_latents_hat = latents_hat[labels==1].mean(axis=1)
            
            # Add all datasets
            datasets.append(normal_latents.numpy())
            category_labels.extend([0] * normal_latents.shape[0])  # normal latents
            
            datasets.append(normal_latents_hat.numpy())
            category_labels.extend([1] * normal_latents_hat.shape[0])  # normal latents_hat
            
            datasets.append(abnormal_latents.numpy())
            category_labels.extend([2] * abnormal_latents.shape[0])  # abnormal latents
            
            datasets.append(abnormal_latents_hat.numpy())
            category_labels.extend([3] * abnormal_latents_hat.shape[0])  # abnormal latents_hat
            
            datasets.append(memory_latents)
            category_labels.extend([4] * memory_latents.shape[0])  # memory vectors
            
        else:
            # Original behavior
            if use_latents_hat:
                latents = latents_hat

            if not use_latents_avg:
                hidden_dim = latents.shape[-1]
                normal_latents = latents[labels==0].view(-1, hidden_dim)
                abnormal_latents = latents[labels==1].view(-1, hidden_dim)
            else:
                normal_latents = latents[labels==0].mean(axis=1)
                abnormal_latents = latents[labels==1].mean(axis=1)
            
            datasets.append(normal_latents.numpy())
            category_labels.extend([0] * normal_latents.shape[0])
            
            datasets.append(abnormal_latents.numpy())
            category_labels.extend([1] * abnormal_latents.shape[0])
            
            datasets.append(memory_latents)
            category_labels.extend([2] * memory_latents.shape[0])
        
        # Combine all data
        combined_data = np.vstack(datasets)
        category_labels = np.array(category_labels)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            metric='euclidean'
        )
        umap_results = reducer.fit_transform(scaled_data)
        
        # Create the plot
        plt.figure(figsize=(10, 8), dpi=200)
        
        # Define colors and markers for each category
        if use_both_latents:
            plot_config = {
                0: {'label': 'Normal Latents', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.6, 's': 80},
                1: {'label': 'Normal Latents_hat', 'color': '#aec7e8', 'marker': 's', 'alpha': 0.6, 's': 80},
                2: {'label': 'Abnormal Latents', 'color': '#d62728', 'marker': '^', 'alpha': 0.6, 's': 80},
                3: {'label': 'Abnormal Latents_hat', 'color': '#ff9896', 'marker': 'v', 'alpha': 0.6, 's': 80},
                4: {'label': 'Memory Vectors', 'color': '#2ca02c', 'marker': 'D', 'alpha': 0.8, 's': 120}
            }
        else:
            plot_config = {
                0: {'label': 'Normal Latents', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.6, 's': 100},
                1: {'label': 'Abnormal Latents', 'color': '#d62728', 'marker': '^', 'alpha': 0.6, 's': 100}, 
                2: {'label': 'Memory Vectors', 'color': '#2ca02c', 'marker': 's', 'alpha': 0.8, 's': 150}
            }
        
        # Plot each category
        for cat_id, config in plot_config.items():
            mask = (category_labels == cat_id)
            if mask.any():
                plt.scatter(
                    umap_results[mask, 0], 
                    umap_results[mask, 1],
                    label=config['label'],
                    c=config['color'],
                    marker=config['marker'],
                    alpha=config['alpha'],
                    s=config['s'],
                    edgecolors='white',
                    linewidth=0.5
                )
        
        plt.title(f'UMAP: {filename} vs Memory Vectors • {self.train_config["dataset_name"].upper()}', 
                fontsize=14, pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f'umap_{filename}_memory.png')

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"UMAP saved into '{out_path}'.")


    def plot_tsne_single_class_with_memory(
        self,
        use_normal: bool = True,  
        use_latents_avg: bool = False,
    ):
        # get latents and memory vector
        latents, latents_hat, labels = self.get_latent()
        
        class_name = 'normal' if use_normal else 'abnormal'
        label_value = 0 if use_normal else 1
        filename = f'{class_name}_latents_avg' if use_latents_avg else f'{class_name}_latents'
        
        # Select data for chosen class
        class_mask = (labels == label_value)
        class_latents = latents[class_mask]
        class_latents_hat = latents_hat[class_mask]
        
        if class_latents.shape[0] == 0:
            print(f"No {class_name} samples found.")
            return
        
        memory_latents = self.model.memory.memories  # (num_memory, D)
        memory_latents = F.normalize(memory_latents, dim=-1)
        memory_latents = memory_latents.detach().cpu().numpy()
        
        datasets = []
        category_labels = []
        
        if not use_latents_avg:
            hidden_dim = class_latents.shape[-1]
            # Flatten to (num_samples * num_latents, hidden_dim)
            class_latents_flat = class_latents.view(-1, hidden_dim)
            class_latents_hat_flat = class_latents_hat.view(-1, hidden_dim)
        else:
            # Average over sequence dimension
            class_latents_flat = class_latents.mean(axis=1)  # (num_samples, hidden_dim)
            class_latents_hat_flat = class_latents_hat.mean(axis=1)
        
        # Add datasets
        datasets.append(class_latents_flat.numpy())
        category_labels.extend([0] * class_latents_flat.shape[0])  # original latents
        
        datasets.append(class_latents_hat_flat.numpy())
        category_labels.extend([1] * class_latents_hat_flat.shape[0])  # reconstructed latents
        
        datasets.append(memory_latents)
        category_labels.extend([2] * memory_latents.shape[0])  # memory vectors
        
        # Combine all data
        combined_data = np.vstack(datasets)
        category_labels = np.array(category_labels)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
        # Apply t-SNE
        n_samples = scaled_data.shape[0]
        perplexity = min(30, max(5, n_samples // 3)) if n_samples > 10 else 5
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            n_iter=1000, 
            init='pca', 
            learning_rate='auto'
        )
        tsne_results = tsne.fit_transform(scaled_data)
        
        # Create the plot
        plt.figure(figsize=(10, 8), dpi=200)
        
        # Define colors and markers for each category
        if use_normal:
            plot_config = {
                0: {'label': 'Normal Latents', 'color': '#1f77b4', 'marker': 'o', 'alpha': 0.7, 's': 80},
                1: {'label': 'Normal Latents_hat', 'color': '#aec7e8', 'marker': 's', 'alpha': 0.7, 's': 80},
                2: {'label': 'Memory Vectors', 'color': '#2ca02c', 'marker': 'D', 'alpha': 0.8, 's': 120}
            }
        else:
            plot_config = {
                0: {'label': 'Abnormal Latents', 'color': '#d62728', 'marker': '^', 'alpha': 0.7, 's': 80},
                1: {'label': 'Abnormal Latents_hat', 'color': '#ff9896', 'marker': 'v', 'alpha': 0.7, 's': 80},
                2: {'label': 'Memory Vectors', 'color': '#2ca02c', 'marker': 'D', 'alpha': 0.8, 's': 120}
            }
        
        # Plot each category
        for cat_id, config in plot_config.items():
            mask = (category_labels == cat_id)
            if mask.any():
                plt.scatter(
                    tsne_results[mask, 0], 
                    tsne_results[mask, 1],
                    label=config['label'],
                    c=config['color'],
                    marker=config['marker'],
                    alpha=config['alpha'],
                    s=config['s'],
                    edgecolors='white',
                    linewidth=0.5
                )
        
        plt.title(f't-SNE: {class_name.title()} Latents vs Memory • {self.train_config["dataset_name"].upper()}', 
                fontsize=14, pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f't-sne_{filename}_with_memory.png')

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Single class t-SNE saved into '{out_path}'.")
        print(f"Samples - {class_name}: {class_latents_flat.shape[0]}, Memory: {memory_latents.shape[0]}")


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

    # def plot_attn_single_self_sum(self):
    #     self.model.eval()

    #     for (X, y) in self.test_loader:
    #         X = X.to(self.device)
    #         loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = self.model(X, return_attn_weight=True)
    #         break
        
    #     single_attn_weight_enc = attn_weight_enc[0, :, :, :] # (H, F, N)
    #     single_attn_weight_dec = attn_weight_dec[0, :, :, :] # (H, F, N)

    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    #     enc_head_avg = single_attn_weight_enc.mean(dim=0).detach().cpu().numpy() # (F, N)
    #     im1 = axes[0].imshow(enc_head_avg, cmap='viridis', aspect='auto')
    #     axes[0].set_title('Encoder Attention')
    #     axes[0].set_xlabel('Column')
    #     axes[0].set_ylabel('Latent')
    #     plt.colorbar(im1, ax=axes[0])

    #     all_self_attn = []
    #     for self_attn in attn_weight_self_list:
    #         single_self_attn = self_attn[0] # 1st
    #         layer_avg = single_self_attn.mean(0) # 
    #         all_self_attn.append(layer_avg)
        
    #     self_attn_avg = torch.stack(all_self_attn).mean(dim=0).detach().cpu().numpy() # (F, N)
    #     im2 = axes[1].imshow(self_attn_avg, cmap='viridis', aspect='auto')
    #     axes[1].set_title('Self Attention (All Layers Avg)')
    #     axes[1].set_xlabel('Latent')
    #     axes[1].set_ylabel('Latent')
    #     plt.colorbar(im2, ax=axes[1])

    #     dec_head_avg = single_attn_weight_dec.mean(dim=0).detach().cpu().numpy() # (F, N)
    #     im3 = axes[2].imshow(dec_head_avg, cmap='viridis', aspect='auto')
    #     axes[2].set_title('Decoder Attention')
    #     axes[2].set_xlabel('Latent')
    #     axes[2].set_ylabel('Column')
    #     plt.colorbar(im3, ax=axes[2])

    #     plt.tight_layout()
        
    #     filename = 'single_sample_enc_self_dec'
    #     base_path = self.train_config['base_path']
    #     out_path = os.path.join(base_path, f'{filename}.png')
    #     plt.savefig(out_path, bbox_inches='tight', dpi=200)
    #     plt.close()
        
    #     print(f"Attention analysis (1x3 plot) saved to '{out_path}'")

    @torch.no_grad()
    def plot_hist_diff_memory_addressing(self):
        """
        Plot histogram of L2 distances between latents before and after memory addressing.
        Shows the difference for normal vs abnormal samples from test_loader.
        """
        self.model.eval()
        
        normal_distances = []
        abnormal_distances = []
        
        for (X, y) in self.test_loader:
            X_input = X.to(self.device)
            _, latents, latents_hat = self.model(X_input, return_latents=True)
            
            # Calculate L2 distance between latents and latents_hat
            # latents: (B, N, D), latents_hat: (B, N, D)
            l2_distances = torch.norm(latents - latents_hat, dim=[1,2])  # (B, N)
            # l2_distances = l2_distances.mean(dim=-1)  # Average over latent dimension -> (B,)
            
            # Separate normal and abnormal
            normal_mask = (y == 0)
            abnormal_mask = (y == 1)
            
            if normal_mask.any():
                normal_distances.append(l2_distances[normal_mask].cpu().numpy())
            if abnormal_mask.any():
                abnormal_distances.append(l2_distances[abnormal_mask].cpu().numpy())
        
        # Concatenate all distances
        if normal_distances:
            normal_distances = np.concatenate(normal_distances, axis=0)
        else:
            normal_distances = np.array([])
            
        if abnormal_distances:
            abnormal_distances = np.concatenate(abnormal_distances, axis=0)
        else:
            abnormal_distances = np.array([])
        
        # Check if we have data
        if normal_distances.size == 0 and abnormal_distances.size == 0:
            print("No data found for histogram plotting.")
            return
        
        # Determine global range for consistent binning
        all_distances = []
        if normal_distances.size > 0:
            all_distances.append(normal_distances)
        if abnormal_distances.size > 0:
            all_distances.append(abnormal_distances)
        
        if len(all_distances) > 0:
            all_distances = np.concatenate(all_distances)
            global_min = all_distances.min()
            global_max = all_distances.max()
            
            # Handle edge case where all distances are the same
            if global_min == global_max:
                eps = 1e-6 if global_min == 0 else abs(global_min) * 1e-3
                global_min -= eps
                global_max += eps
            
            bins = np.linspace(global_min, global_max, 50)
        else:
            bins = 50
        
        # Create single plot
        plt.figure(figsize=(10, 6), dpi=200)
        
        if normal_distances.size > 0:
            plt.hist(normal_distances, bins=bins, alpha=0.6, density=True, 
                    label=f'Normal (n={len(normal_distances)})', color='#1f77b4')
        if abnormal_distances.size > 0:
            plt.hist(abnormal_distances, bins=bins, alpha=0.6, density=True, 
                    label=f'Abnormal (n={len(abnormal_distances)})', color='#d62728')
        
        plt.xlabel('L2 Distance (latents vs latents_hat)')
        plt.ylabel('Density')
        plt.title(f'Memory Addressing L2 Distance • {self.train_config["dataset_name"].upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, 'hist_memory_addressing_l2_distance.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        # Print statistics
        if normal_distances.size > 0:
            print(f"Normal samples L2 distance - Mean: {normal_distances.mean():.4f}, Std: {normal_distances.std():.4f}")
        if abnormal_distances.size > 0:
            print(f"Abnormal samples L2 distance - Mean: {abnormal_distances.mean():.4f}, Std: {abnormal_distances.std():.4f}")
        
        print(f"Memory addressing L2 distance histogram saved to '{out_path}'.")

    @torch.no_grad()
    def get_single_samples(self, num_samples: int = 1):
        """Normal과 Abnormal 샘플 각각 하나씩 logger로 출력"""
        self.model.eval()
        
        collected_data = {
            'normal': {
                'x': [], 'x_hat': [], 'latents': [], 'latents_hat': [], 'memory_weights': [], 'losses': []
            },
            'abnormal': {
                'x': [], 'x_hat': [], 'latents': [], 'latents_hat': [], 'memory_weights': [], 'losses': []
            }
        }
        
        normal_count = 0
        abnormal_count = 0
        
        for (X, y) in self.test_loader:
            if normal_count >= num_samples and abnormal_count >= num_samples:
                break
                
            X_input = X.to(self.device)
            
            loss, latents, latents_hat = self.model(X_input, return_latents=True)
            _, memory_weight = self.model(X_input, return_memory_weight=True)
            _, x_orig, x_hat = self.model(X_input, return_pred=True)
            
            normal_mask = (y == 0)
            if normal_mask.any() and normal_count < num_samples:
                needed = min(num_samples - normal_count, normal_mask.sum().item())
                normal_indices = torch.where(normal_mask)[0][:needed]
                
                collected_data['normal']['x'].append(x_orig[normal_indices].cpu().numpy())
                collected_data['normal']['x_hat'].append(x_hat[normal_indices].cpu().numpy())
                collected_data['normal']['latents'].append(latents[normal_indices].cpu().numpy())
                collected_data['normal']['latents_hat'].append(latents_hat[normal_indices].cpu().numpy())
                collected_data['normal']['memory_weights'].append(memory_weight[normal_indices].cpu().numpy())
                collected_data['normal']['losses'].append(loss[normal_indices].cpu().numpy())
                
                normal_count += needed
            
            abnormal_mask = (y == 1)
            if abnormal_mask.any() and abnormal_count < num_samples:
                needed = min(num_samples - abnormal_count, abnormal_mask.sum().item())
                abnormal_indices = torch.where(abnormal_mask)[0][:needed]
                
                collected_data['abnormal']['x'].append(x_orig[abnormal_indices].cpu().numpy())
                collected_data['abnormal']['x_hat'].append(x_hat[abnormal_indices].cpu().numpy())
                collected_data['abnormal']['latents'].append(latents[abnormal_indices].cpu().numpy())
                collected_data['abnormal']['latents_hat'].append(latents_hat[abnormal_indices].cpu().numpy())
                collected_data['abnormal']['memory_weights'].append(memory_weight[abnormal_indices].cpu().numpy())
                collected_data['abnormal']['losses'].append(loss[abnormal_indices].cpu().numpy())
                
                abnormal_count += needed
        
        final_data = {}
        for data_type in ['normal', 'abnormal']:
            if collected_data[data_type]['x']:  
                final_data[data_type] = {}
                for key in collected_data[data_type]:
                    final_data[data_type][key] = np.concatenate(collected_data[data_type][key], axis=0)[:num_samples]
        
        # Logger로 단일 샘플 출력
        self.log_single_sample_analysis(final_data, num_samples)
        
        return final_data

    def log_single_sample_analysis(self, data, num_samples):
        """Logger를 통해 각 샘플의 상세 정보 출력"""
        
        self.logger.info("=" * 60)
        self.logger.info(f"SINGLE SAMPLE ANALYSIS (n={num_samples} each)")
        self.logger.info("=" * 60)
        
        # Memory vectors 출력
        memory_vectors = self.model.memory.memories.detach().cpu().numpy()
        self.logger.info(f"\nMemory vectors shape: {memory_vectors.shape}")
        self.logger.info(f"Memory vectors:\n{memory_vectors}")
        
        for data_type in ['normal', 'abnormal']:
            if data_type not in data:
                continue
                
            self.logger.info(f"\n--- {data_type.upper()} SAMPLES ---")
            
            for i in range(min(num_samples, len(data[data_type]['x']))):
                self.logger.info(f"\n[{data_type.upper()} Sample {i+1}]")
                
                # 1. 기본 정보
                x = data[data_type]['x'][i]
                x_hat = data[data_type]['x_hat'][i]
                loss = data[data_type]['losses'][i]
                
                self.logger.info(f"Original data (x): {x}")
                self.logger.info(f"Reconstructed (x_hat): {x_hat}")
                self.logger.info(f"Loss: {loss:.6f}")
                self.logger.info(f"Feature-wise MSE: {(x - x_hat)**2}")
                
                # 2. Latent 분석
                latents = data[data_type]['latents'][i]  # (N, D)
                latents_hat = data[data_type]['latents_hat'][i]  # (N, D)
                
                self.logger.info(f"Latents (z) shape: {latents.shape}")
                self.logger.info(f"Latents (z):\n{latents}")
                self.logger.info(f"Latents_hat (ẑ):\n{latents_hat}")
                
                # L2 norm 비교
                latents_norm = np.linalg.norm(latents, axis=-1)  # (N,)
                latents_hat_norm = np.linalg.norm(latents_hat, axis=-1)  # (N,)
                
                self.logger.info(f"Latent L2 norms (z): {latents_norm}")
                self.logger.info(f"Latent L2 norms (ẑ): {latents_hat_norm}")
                
                # 정규화 후 코사인 유사도
                latents_unit = latents / (np.linalg.norm(latents, axis=-1, keepdims=True) + 1e-8)
                latents_hat_unit = latents_hat / (np.linalg.norm(latents_hat, axis=-1, keepdims=True) + 1e-8)
                cosine_sim = np.sum(latents_unit * latents_hat_unit, axis=-1)  # (N,)
                
                self.logger.info(f"Cosine similarity (z, ẑ): {cosine_sim}")
                
                # 3. Memory weight 분석
                memory_weights = data[data_type]['memory_weights'][i]  # (N, M)
                
                self.logger.info(f"Memory weights shape: {memory_weights.shape}")
                self.logger.info(f"Memory weights:\n{memory_weights}")
                
                # 각 latent별 top-3 memory 사용
                for latent_idx in range(memory_weights.shape[0]):
                    top_memories = np.argsort(memory_weights[latent_idx])[-3:][::-1]
                    top_weights = memory_weights[latent_idx][top_memories]
                    self.logger.info(f"Latent {latent_idx} top-3 memories: slots {top_memories} with weights {top_weights}")
                
                # Memory usage entropy
                entropy = -np.sum(memory_weights * np.log(memory_weights + 1e-8), axis=-1)
                self.logger.info(f"Memory entropy per latent: {entropy}")
                
                # Memory addressing을 통한 변화량
                latent_diff = np.linalg.norm(latents - latents_hat, axis=-1)
                self.logger.info(f"L2 distance (z - ẑ) per latent: {latent_diff}")
                
                self.logger.info("-" * 40)
        
        # 4. 전체 비교 (둘 다 있는 경우)
        if 'normal' in data and 'abnormal' in data and len(data['normal']['losses']) > 0 and len(data['abnormal']['losses']) > 0:
            self.logger.info(f"\nNORMAL vs ABNORMAL COMPARISON")
            
            # Loss 비교
            normal_losses = data['normal']['losses']
            abnormal_losses = data['abnormal']['losses']
            self.logger.info(f"Average Loss - Normal: {np.mean(normal_losses):.6f}, Abnormal: {np.mean(abnormal_losses):.6f}")
            
            # Memory entropy 비교
            normal_entropy = []
            abnormal_entropy = []
            
            for i in range(len(data['normal']['memory_weights'])):
                weights = data['normal']['memory_weights'][i]
                entropy = -np.sum(weights * np.log(weights + 1e-8), axis=-1)
                normal_entropy.extend(entropy)
                
            for i in range(len(data['abnormal']['memory_weights'])):
                weights = data['abnormal']['memory_weights'][i]
                entropy = -np.sum(weights * np.log(weights + 1e-8), axis=-1)
                abnormal_entropy.extend(entropy)
                
            self.logger.info(f"Average Memory Entropy - Normal: {np.mean(normal_entropy):.4f}, Abnormal: {np.mean(abnormal_entropy):.4f}")
            
            # Cosine similarity 비교
            normal_cosine = []
            abnormal_cosine = []
            
            for i in range(len(data['normal']['latents'])):
                latents = data['normal']['latents'][i]
                latents_hat = data['normal']['latents_hat'][i]
                latents_unit = latents / (np.linalg.norm(latents, axis=-1, keepdims=True) + 1e-8)
                latents_hat_unit = latents_hat / (np.linalg.norm(latents_hat, axis=-1, keepdims=True) + 1e-8)
                cosine_sim = np.sum(latents_unit * latents_hat_unit, axis=-1)
                normal_cosine.extend(cosine_sim)
            
            for i in range(len(data['abnormal']['latents'])):
                latents = data['abnormal']['latents'][i]
                latents_hat = data['abnormal']['latents_hat'][i]
                latents_unit = latents / (np.linalg.norm(latents, axis=-1, keepdims=True) + 1e-8)
                latents_hat_unit = latents_hat / (np.linalg.norm(latents_hat, axis=-1, keepdims=True) + 1e-8)
                cosine_sim = np.sum(latents_unit * latents_hat_unit, axis=-1)
                abnormal_cosine.extend(cosine_sim)
                
            self.logger.info(f"Average Cosine Similarity - Normal: {np.mean(normal_cosine):.4f}, Abnormal: {np.mean(abnormal_cosine):.4f}")
        
        self.logger.info("=" * 60)

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
    def plot_attn_dec_memory(self, sample_idx: int = 0):
        """
        Plot decoder attention maps for normal/abnormal samples using latents vs latents_hat
        Creates a 2x2 plot showing:
        - Top left: Normal sample with latents
        - Top right: Normal sample with latents_hat
        - Bottom left: Abnormal sample with latents
        - Bottom right: Abnormal sample with latents_hat
        
        Args:
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

        def get_decoder_attention(X_batch, use_latents_hat=False):
            """Get decoder attention map for a sample"""
            X_batch = X_batch.to(self.device)
            
            # Get latents and latents_hat
            _, latents, latents_hat = self.model(X_batch, return_latents=True)
            
            # Choose which latents to use
            if use_latents_hat:
                memory_input = latents_hat[0:1]  # (1, N, D)
            else:
                memory_input = latents[0:1]  # (1, N, D)
            
            # Get decoder query with positional encoding
            decoder_query = self.model.decoder_query + self.model.pos_encoding  # (1, F, D)
            
            # Forward through decoder to get attention weights
            # We need to manually call the decoder's attention mechanism
            # This assumes the decoder has an attention mechanism that returns weights
            
            # If the decoder uses multi-head attention, we'll need to access it directly
            # For now, let's assume we can get attention from the model's forward pass
            with torch.no_grad():
                # Get attention weights from decoder
                loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = \
                    self.model(X_batch, return_attn_weight=True)
                
                # Decoder attention: (B, H, F, N) - from features to latents
                dec_attn = attn_weight_dec[0, :, :, :].mean(dim=0)  # Average over heads: (F, N)
                
            return dec_attn.detach().cpu().numpy()

        # Find normal and abnormal samples
        X_normal, y_normal = find_sample_by_label(self.test_loader, target_label=0, sample_idx=sample_idx)
        X_abnormal, y_abnormal = find_sample_by_label(self.test_loader, target_label=1, sample_idx=sample_idx)
        
        if X_normal is None or X_abnormal is None:
            print(f"Warning: Could not find required samples at index {sample_idx}")
            return None
        
        # Get all 4 attention maps
        normal_latents_attn = get_decoder_attention(X_normal, use_latents_hat=False)
        normal_latents_hat_attn = get_decoder_attention(X_normal, use_latents_hat=True)
        abnormal_latents_attn = get_decoder_attention(X_abnormal, use_latents_hat=False)
        abnormal_latents_hat_attn = get_decoder_attention(X_abnormal, use_latents_hat=True)
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
        
        # Plot configurations
        plots = [
            (normal_latents_attn, f'Normal (latents)\nLabel: {y_normal[0].item()}', axes[0, 0]),
            (normal_latents_hat_attn, f'Normal (latents_hat)\nLabel: {y_normal[0].item()}', axes[0, 1]),
            (abnormal_latents_attn, f'Abnormal (latents)\nLabel: {y_abnormal[0].item()}', axes[1, 0]),
            (abnormal_latents_hat_attn, f'Abnormal (latents_hat)\nLabel: {y_abnormal[0].item()}', axes[1, 1])
        ]
        
        # Find global min/max for consistent color scaling
        all_attns = [normal_latents_attn, normal_latents_hat_attn, 
                    abnormal_latents_attn, abnormal_latents_hat_attn]
        vmin = min(attn.min() for attn in all_attns)
        vmax = max(attn.max() for attn in all_attns)
        
        # Plot each attention map
        for attn_data, title, ax in plots:
            im = ax.imshow(attn_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Latent Index')
            ax.set_ylabel('Feature Index')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Overall title
        fig.suptitle(f'Decoder Attention Maps: Latents vs Latents_hat • {self.train_config["dataset_name"].upper()}', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        # Save the plot
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f'decoder_attention_memory_comparison_idx{sample_idx}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"Decoder attention memory comparison saved to '{out_path}'")
        print(f"Normal sample label: {y_normal[0].item()}, Abnormal sample label: {y_abnormal[0].item()}")
        
        return out_path


    @torch.no_grad()
    def plot_attn_pair(self, sample_idx: int = 0, abnormal_idx: int = None):
        """
        Create a 2x3 plot showing attention maps for normal and abnormal samples
        Row 0: Normal sample (encoder, self, decoder)
        Row 1: Abnormal sample (encoder, self, decoder)
        """
        self.model.eval()

        if abnormal_idx is None:
            abnormal_idx = sample_idx

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

        def get_attention_maps(X_batch):
            """Get averaged attention maps for a sample"""
            X_batch = X_batch.to(self.device)
            loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec = \
                self.model(X_batch, return_attn_weight=True)
            
            # Average over heads and select first sample
            enc_attn = attn_weight_enc[0].mean(dim=0).detach().cpu().numpy()  # (F, N)
            dec_attn = attn_weight_dec[0].mean(dim=0).detach().cpu().numpy()  # (F, N)
            
            # Average self attention over both layers and heads
            if attn_weight_self_list:
                all_self_attn = torch.stack([self_attn[0] for self_attn in attn_weight_self_list], dim=0)  # (Depth, H, N, N)
                self_attn = all_self_attn.mean(dim=(0, 1)).detach().cpu().numpy()  # (N, N)
            else:
                # Fallback to identity if no self attention
                N = enc_attn.shape[1]
                self_attn = np.eye(N)
            
            return enc_attn, self_attn, dec_attn

        # Find normal and abnormal samples
        X_normal, y_normal = find_sample_by_label(self.test_loader, target_label=0, sample_idx=sample_idx)
        X_abnormal, y_abnormal = find_sample_by_label(self.test_loader, target_label=1, sample_idx=abnormal_idx)
        
        if X_normal is None or X_abnormal is None:
            print(f"Warning: Could not find required samples at index {sample_idx}")
            return None
        
        # Get attention maps for both samples
        normal_enc, normal_self, normal_dec = get_attention_maps(X_normal)
        abnormal_enc, abnormal_self, abnormal_dec = get_attention_maps(X_abnormal)
        
        # Create 2x3 subplot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=200)
        
        # Plot configurations: (data, title, colormap)
        plots = [
            # Row 0: Normal sample
            (normal_enc, f'Normal Encoder\n(Label: {y_normal[0].item()})', 'viridis'),
            (normal_self, 'Normal Self-Attention', 'viridis'),
            (normal_dec, 'Normal Decoder', 'viridis'),
            # Row 1: Abnormal sample  
            (abnormal_enc, f'Abnormal Encoder\n(Label: {y_abnormal[0].item()})', 'viridis'),
            (abnormal_self, 'Abnormal Self-Attention', 'viridis'),
            (abnormal_dec, 'Abnormal Decoder', 'viridis')
        ]
        
        # Find global min/max for consistent color scaling within each attention type
        enc_vmin, enc_vmax = min(normal_enc.min(), abnormal_enc.min()), max(normal_enc.max(), abnormal_enc.max())
        self_vmin, self_vmax = min(normal_self.min(), abnormal_self.min()), max(normal_self.max(), abnormal_self.max())
        dec_vmin, dec_vmax = min(normal_dec.min(), abnormal_dec.min()), max(normal_dec.max(), abnormal_dec.max())
        
        # Color scale ranges for each column
        v_ranges = [
            (enc_vmin, enc_vmax),   # Encoder column
            (self_vmin, self_vmax), # Self-attention column
            (dec_vmin, dec_vmax)    # Decoder column
        ]
        
        # Plot each attention map
        for idx, (attn_data, title, cmap) in enumerate(plots):
            row, col = idx // 3, idx % 3
            vmin, vmax = v_ranges[col]
            
            im = axes[row, col].imshow(attn_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            axes[row, col].set_title(title, fontsize=12)
            
            # Set appropriate axis labels based on attention type
            if col == 0:  # Encoder
                axes[row, col].set_xlabel('Latent Index')
                axes[row, col].set_ylabel('Feature Index')
            elif col == 1:  # Self-attention
                axes[row, col].set_xlabel('Latent Index')
                axes[row, col].set_ylabel('Latent Index')
            else:  # Decoder
                axes[row, col].set_xlabel('Latent Index')
                axes[row, col].set_ylabel('Feature Index')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            
            # Add colorbar to each subplot
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Overall title
        fig.suptitle(f'Attention Maps Comparison (Sample {sample_idx}) • {self.train_config["dataset_name"].upper()}', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save the plot
        base_path = self.train_config['base_path']
        out_path = os.path.join(base_path, f'attention_pair_comparison_idx{sample_idx}_{abnormal_idx}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"Attention pair comparison saved to '{out_path}'")
        print(f"Normal sample label: {y_normal[0].item()}, Abnormal sample label: {y_abnormal[0].item()}")
        
        return out_path 