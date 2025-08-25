import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from DataSet.DataLoader import get_dataloader
from models.Perceiver.Model import Perceiver
from utils import aucPerformance, F1Performance


class Analyzer(object):
    def __init__(self, model_config: dict):
        self.train_loader, self.test_loader = get_dataloader(model_config)
        self.device = model_config['device']
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = Perceiver(
            num_features=model_config['data_dim'],
            num_heads=model_config['num_heads'],
            depth=model_config['depth'],
            hidden_dim=model_config['hidden_dim'],
            mlp_ratio=model_config['mlp_ratio'],
            dropout_prob=model_config['dropout_prob'],
            drop_col_prob=model_config['drop_col_prob'],
        ).to(self.device)
        self.logger = model_config['logger']
        self.model_config = model_config
        self.epochs = model_config['epochs']

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


    def training(self):
        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        num_features = self.model_config['data_dim']
        self_attention_map  = torch.zeros(num_features, num_features, device=self.device)
        cross_attention_map = torch.zeros(num_features, num_features, device=self.device)
        self_count_map      = torch.zeros(num_features, num_features, device=self.device)
        cross_count_map     = torch.zeros(num_features, num_features, device=self.device)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                losses, attns, input_col_idx, target_col_idx = self.model(x_input, return_weight=True)
                loss = losses.mean() # (B) -> scalar
                # loss = self.model(x_input).mean() # (B) -> scalar
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self._accumulate_attn(
                    attns, input_col_idx, target_col_idx,
                    self_attention_map, self_count_map,
                    cross_attention_map, cross_count_map
                )

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            self.logger.info(info.format(epoch,loss.cpu()))
        print("Training complete.")

        self._finalize_and_save_attention_maps(
            self_attention_map, self_count_map,
            cross_attention_map, cross_count_map
        )

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