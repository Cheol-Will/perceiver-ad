import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.AutoEncoder.Trainer import Trainer

class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict, analysis_config: dict):
        super().__init__(model_config, train_config)
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

    def plot_grad_z_x(self, sample_idx=0):
        """
        Visualize gradient of latent z with respect to input x
        Shows average gradient for normal samples vs individual samples
        
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

        def compute_gradient_z_x(X_batch):
            """
            Compute gradient of latent z with respect to input x
            Returns gradient matrix of shape (latent_dim, input_dim)
            """
            X_batch = X_batch.to(self.device)
            X_batch.requires_grad_(True) # need grad on
            
            # Forward pass to get latent representation
            loss, x, z, x_hat = self.model(X_batch, return_analysis=True)
            
            # Compute gradient dz/dx for each latent dimension
            gradients = []
            for i in range(z.shape[1]):  # For each latent dimension
                # Sum over batch dimension to get average gradient
                z_i = z[:, i].sum()  # Sum over batch
                grad = torch.autograd.grad(z_i, X_batch, retain_graph=True, create_graph=False)[0]
                gradients.append(grad.mean(dim=0))  # Average over batch
            
            gradient_matrix = torch.stack(gradients, dim=0)  # (latent_dim, input_dim)
            return gradient_matrix.detach().cpu().numpy()

        def compute_single_gradient_z_x(X_batch):
            """
            Compute gradient for a single sample
            """
            X_batch = X_batch.to(self.device)
            X_batch.requires_grad_(True)
            
            loss, x, z, x_hat = self.model(X_batch, return_analysis=True)
            
            gradients = []
            for i in range(z.shape[1]):  
                z_i = z[0, i]  
                grad = torch.autograd.grad(z_i, X_batch, retain_graph=True, create_graph=False)[0]
                gradients.append(grad[0])  
            
            gradient_matrix = torch.stack(gradients, dim=0)  # (latent_dim, input_dim)
            return gradient_matrix.detach().cpu().numpy()

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

        # Get normal average gradient
        normal_avg_gradient = compute_gradient_z_x(X_normal_all)

        # Collect individual samples
        individual_data = []
        for idx in sample_indices:
            X_single, y_single = find_sample_by_label(self.test_loader, target_label=0, sample_idx=idx)
            if X_single is None:
                X_single, y_single = find_sample_by_label(self.train_loader, target_label=0, sample_idx=idx)
            
            if X_single is None:
                raise RuntimeError(f"Could not find sample at index {idx}")
            
            single_gradient = compute_single_gradient_z_x(X_single)
            individual_data.append({
                'gradient': single_gradient,
                'label': y_single[0].item(),
                'idx': idx
            })

        # Setup plot dimensions
        num_rows = 1 + len(sample_indices)
        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows), dpi=200)
        
        if num_rows == 1:
            axes = [axes]

        # Calculate global color ranges
        all_gradients = [normal_avg_gradient]
        for data in individual_data:
            all_gradients.append(data['gradient'])
        
        global_vmin = min(grad.min() for grad in all_gradients)
        global_vmax = max(grad.max() for grad in all_gradients)

        # Plot normal average (row 0)
        im = axes[0].imshow(normal_avg_gradient, cmap='RdBu_r', aspect='auto', 
                        vmin=global_vmin, vmax=global_vmax)
        axes[0].set_xlabel('Input Features')
        axes[0].set_ylabel('Latent Dimensions')
        axes[0].set_title(f'Gradient of z with respect to x (Normal Samples, n={len(X_normal_all)})')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot individual samples (remaining rows)
        for row_idx, data in enumerate(individual_data, 1):
            im = axes[row_idx].imshow(data['gradient'], cmap='RdBu_r', aspect='auto',
                                    vmin=global_vmin, vmax=global_vmax)
            axes[row_idx].set_xlabel('Input Features')
            axes[row_idx].set_ylabel('Latent Dimensions')
            axes[row_idx].set_title(f'Gradient of z with respect to x (Sample {data["idx"]}, Label: {data["label"]})')
            axes[row_idx].set_xticks([])
            axes[row_idx].set_yticks([])
            plt.colorbar(im, ax=axes[row_idx], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save plot
        base_path = self.train_config['base_path']
        sample_filename = "_".join(map(str, sample_indices)) if len(sample_indices) > 1 else str(sample_indices[0])
        out_path = os.path.join(base_path, f'gradient_z_x_{num_rows}x1_samples_{sample_filename}_{self.train_config["dataset_name"]}.pdf')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        out_path = os.path.join(base_path, f'gradient_z_x_{num_rows}x1_samples_{sample_filename}_{self.train_config["dataset_name"]}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"Gradient visualization saved to '{out_path}'")
        print(f"Normal samples averaged: {len(X_normal_all)}")
        print(f"Gradient matrix shape: {normal_avg_gradient.shape} (latent_dim x input_dim)")
        for data in individual_data:
            print(f"Sample {data['idx']} label: {data['label']}")
        
        return out_path