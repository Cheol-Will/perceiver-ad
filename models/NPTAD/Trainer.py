import os
import torch
import torch.optim as optim
import numpy as np
from DataSet.DataLoader import get_dataloader
from models.NPTAD.Model import NPTAD
from utils import aucPerformance, F1Performance
import math

def nearest_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        self.device = train_config['device']
        self.model = NPTAD(**model_config).to(self.device)

        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        
        self.num_context_samples = train_config.get('num_context_samples', 100)
        self.use_mask_bank = train_config.get('use_mask_bank', True)

        self.model_config = model_config
        self.train_config = train_config

    def _cache_train_data(self):
        """Cache entire training dataset for use as support set"""
        all_train_data = []
        all_train_labels = []
        
        for x_batch, y_batch in self.train_loader:
            all_train_data.append(x_batch)
            all_train_labels.append(y_batch)
        
        self.train_data_full = torch.cat(all_train_data, dim=0)
        self.train_labels_full = torch.cat(all_train_labels, dim=0)
        
        self.logger.info(f"Cached full training data: {self.train_data_full.shape}")

    def _sample_support_set(self, num_samples=None):
        """Randomly sample support set from training data"""
        if num_samples is None:
            num_samples = self.num_context_samples
        
        N_train = len(self.train_data_full)
        num_samples = min(num_samples, N_train)
        
        indices = torch.randperm(N_train)[:num_samples]
        support_data = self.train_data_full[indices]
        
        return support_data

    def training(self):
        print(self.model_config)
        print(self.train_config)
        
        self._cache_train_data()

        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        
        self.model.train()
        print("Training Start.")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, (x_query, y_label) in enumerate(self.train_loader):
                x_query = x_query.to(self.device)
                x_support = self._sample_support_set().to(self.device)
                
                X_combined = torch.cat([x_support, x_query], dim=0)
                query_indices = torch.arange(
                    len(x_support), 
                    len(x_support) + len(x_query),
                    device=self.device
                )
                
                loss = self.model(
                    X_combined, 
                    mode='train', 
                    query_indices=query_indices
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            info = f'Epoch: [{epoch+1}/{self.epochs}]\t Loss: {avg_loss:.6f}'
            self.logger.info(info)
            print(info)
            
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self):
        """Evaluation with mask bank"""
        self.model.eval()
        
        if not hasattr(self, 'train_data_full'):
            self._cache_train_data()
            
        x_support = self.train_data_full.to(self.device)
        num_masks = len(self.model.mask_bank)
        
        print(f"Evaluation: {len(x_support)} support samples, {num_masks} masks")
        
        all_scores = []
        all_labels = []
        
        for batch_idx, (x_test, y_test) in enumerate(self.test_loader):
            x_test = x_test.to(self.device)
            
            if self.use_mask_bank:
                batch_scores = []
                
                for mask_idx in range(num_masks):
                    X_combined = torch.cat([x_support, x_test], dim=0)
                    query_indices = torch.arange(
                        len(x_support),
                        len(x_support) + len(x_test),
                        device=self.device
                    )
                    
                    scores = self.model(
                        X_combined,
                        mode='inference',
                        query_indices=query_indices,
                        mask_idx=mask_idx
                    )
                    batch_scores.append(scores)
                
                batch_scores = torch.stack(batch_scores)
                avg_scores = batch_scores.mean(dim=0)
                
            else:
                batch_scores = []
                for _ in range(self.model.num_reconstructions):
                    X_combined = torch.cat([x_support, x_test], dim=0)
                    query_indices = torch.arange(
                        len(x_support),
                        len(x_support) + len(x_test),
                        device=self.device
                    )
                    
                    scores = self.model(
                        X_combined,
                        mode='inference',
                        query_indices=query_indices
                    )
                    batch_scores.append(scores)
                
                batch_scores = torch.stack(batch_scores)
                avg_scores = batch_scores.mean(dim=0)
            
            all_scores.append(avg_scores.cpu())
            all_labels.append(y_test)
        
        scores = torch.cat(all_scores, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        rauc, ap = aucPerformance(scores, labels)
        f1 = F1Performance(scores, labels)
        
        return rauc, ap, f1