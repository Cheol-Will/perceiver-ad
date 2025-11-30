import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from DataSet.DataLoader import get_dataloader
from models.NPTAD.Model import NPTAD
from utils import aucPerformance, F1Performance
import math

from torch.cuda.amp import autocast, GradScaler

def nearest_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        
        # DDP
        self.use_ddp = train_config.get('use_ddp', True)  
        if self.use_ddp:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = train_config['device']

        train_config['use_ddp'] = self.use_ddp
        
        self.train_loader, self.test_loader = get_dataloader(train_config)
        
        self.model = NPTAD(**model_config).to(self.device)
        
        # DDP wrapper
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        
        self.num_context_samples = train_config.get('num_context_samples', 50)
        self.use_mask_bank = train_config.get('use_mask_bank', True)
        
        # ✅ Evaluation용 support set 최대 크기
        self.max_eval_support = train_config.get('max_eval_support', 500)

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
        
        if not self.use_ddp or self.local_rank == 0:
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
        scaler = GradScaler()  # 

        if not self.use_ddp or self.local_rank == 0:
            print(self.model_config)
            print(self.train_config)
        
        print("Caching train data")
        self._cache_train_data()

        if not self.use_ddp or self.local_rank == 0:
            print(f"Before optimizer: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)

        if not self.use_ddp or self.local_rank == 0:
            print(f"After optimizer: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

        self.model.train()
        if not self.use_ddp or self.local_rank == 0:
            print("Training Start.")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_loader_iter = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=(self.use_ddp and self.local_rank != 0)
            )
            
            for step, (x_query, y_label) in enumerate(train_loader_iter):
                x_query = x_query.to(self.device)
                x_support = self._sample_support_set().to(self.device)
                
                X_combined = torch.cat([x_support, x_query], dim=0)
                query_indices = torch.arange(
                    len(x_support), 
                    len(x_support) + len(x_query),
                    device=self.device
                )
                
                with autocast():
                    loss = self.model(
                        X_combined, 
                        mode='train', 
                        query_indices=query_indices
                    )
                
                optimizer.zero_grad()
                
                # ✅ GradScaler 사용
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # ✅ tqdm postfix 업데이트
                if not self.use_ddp or self.local_rank == 0:
                    train_loader_iter.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'avg_loss': f'{epoch_loss/num_batches:.6f}'
                    })
                
            scheduler.step()
            
            # 모든 GPU의 loss 평균 계산
            if self.use_ddp:
                avg_loss = torch.tensor(epoch_loss / num_batches).to(self.device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                avg_loss = avg_loss.item()
            else:
                avg_loss = epoch_loss / num_batches
            
            if not self.use_ddp or self.local_rank == 0:
                info = f'Epoch: [{epoch+1}/{self.epochs}]\t Loss: {avg_loss:.6f}'
                self.logger.info(info)
                print(info)
            
        if not self.use_ddp or self.local_rank == 0:
            print("Training complete.")
        
        if self.use_ddp:
            dist.barrier()
    
  
  
    @torch.no_grad()
    def evaluate(self):
        """Evaluation with sampled support set (max 50,000 samples)"""
        self.model.eval()
        
        if not hasattr(self, 'train_data_full'):
            self._cache_train_data()
        
        model = self.model.module if self.use_ddp else self.model
        num_masks = len(model.mask_bank) if self.use_mask_bank else model.num_reconstructions
        
        total_train = len(self.train_data_full)
        eval_support_size = min(self.max_eval_support, total_train)
        
        if not self.use_ddp or self.local_rank == 0:
            print(f"Evaluation: Using {eval_support_size} / {total_train} training samples as support set")
        
        # ✅ Support set 샘플링
        indices = torch.randperm(total_train)[:eval_support_size]
        x_support = self.train_data_full[indices].to(self.device)
        
        # ✅ 메모리 해제: 이제 필요 없는 train_data_full 삭제
        del self.train_data_full
        del self.train_labels_full
        if hasattr(self, 'train_data_full'):
            delattr(self, 'train_data_full')
        if hasattr(self, 'train_labels_full'):
            delattr(self, 'train_labels_full')
        torch.cuda.empty_cache()
        
        if not self.use_ddp or self.local_rank == 0:
            print(f"GPU memory after support set: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
        
        all_scores = []
        all_labels = []
        
        test_loader = tqdm(self.test_loader, disable=(self.use_ddp and self.local_rank != 0))
        
        for batch_idx, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.to(self.device)
            
            # Query 분산 (DDP)
            if self.use_ddp:
                batch_size = len(x_test)
                chunk_size = (batch_size + self.world_size - 1) // self.world_size
                start_idx = self.local_rank * chunk_size
                end_idx = min(start_idx + chunk_size, batch_size)
                x_test_chunk = x_test[start_idx:end_idx]
                has_data = len(x_test_chunk) > 0
            else:
                x_test_chunk = x_test
                has_data = True
            
            if self.use_mask_bank:
                batch_scores = []
                
                for mask_idx in range(num_masks):
                    if has_data:
                        X_combined = torch.cat([x_support, x_test_chunk], dim=0)
                        query_indices = torch.arange(
                            len(x_support),
                            len(x_support) + len(x_test_chunk),
                            device=self.device
                        )
                        
                        scores = self.model(
                            X_combined,
                            mode='inference',
                            query_indices=query_indices,
                            mask_idx=mask_idx
                        )
                        batch_scores.append(scores)
                        
                        del X_combined
                    else:
                        batch_scores.append(torch.tensor([], device=self.device))
                
                if has_data:
                    batch_scores = torch.stack(batch_scores)
                    avg_scores = batch_scores.mean(dim=0)
                else:
                    avg_scores = torch.tensor([], device=self.device)
            else:
                batch_scores = []
                
                for _ in range(model.num_reconstructions):
                    if has_data:
                        X_combined = torch.cat([x_support, x_test_chunk], dim=0)
                        query_indices = torch.arange(
                            len(x_support),
                            len(x_support) + len(x_test_chunk),
                            device=self.device
                        )
                        
                        scores = self.model(
                            X_combined,
                            mode='inference',
                            query_indices=query_indices
                        )
                        batch_scores.append(scores)
                        
                        del X_combined
                    else:
                        batch_scores.append(torch.tensor([], device=self.device))
                
                if has_data:
                    batch_scores = torch.stack(batch_scores)
                    avg_scores = batch_scores.mean(dim=0)
                else:
                    avg_scores = torch.tensor([], device=self.device)
            
            # DDP all_gather
            if self.use_ddp:
                chunk_sizes = torch.tensor([len(avg_scores)], dtype=torch.long, device=self.device)
                all_chunk_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
                dist.all_gather(all_chunk_sizes, chunk_sizes)
                
                max_chunk_size = max([s.item() for s in all_chunk_sizes])
                
                if len(avg_scores) < max_chunk_size:
                    pad_size = max_chunk_size - len(avg_scores)
                    avg_scores = torch.cat([
                        avg_scores, 
                        torch.zeros(pad_size, device=self.device)
                    ])
                
                gathered_scores = [torch.zeros(max_chunk_size, device=self.device) 
                                for _ in range(self.world_size)]
                dist.all_gather(gathered_scores, avg_scores)
                
                valid_scores = []
                for i, scores in enumerate(gathered_scores):
                    valid_len = all_chunk_sizes[i].item()
                    if valid_len > 0:
                        valid_scores.append(scores[:valid_len])
                
                if len(valid_scores) > 0:
                    avg_scores = torch.cat(valid_scores)
                else:
                    avg_scores = torch.tensor([], device=self.device)
            
            all_scores.append(avg_scores.cpu())
            all_labels.append(y_test)
        
        scores = torch.cat(all_scores, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        if self.use_ddp:
            if self.local_rank == 0:
                rauc, ap = aucPerformance(scores, labels)
                f1 = F1Performance(scores, labels)
                return rauc, ap, f1
            else:
                return None, None, None
        else:
            rauc, ap = aucPerformance(scores, labels)
            f1 = F1Performance(scores, labels)
            return rauc, ap, f1