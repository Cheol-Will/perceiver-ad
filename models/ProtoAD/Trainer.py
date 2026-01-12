import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from DataSet.DataLoader import get_mq_dataloader
from models.ProtoAD.Model import PrototypeAD
from utils import aucPerformance, F1Performance
import copy


class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_mq_dataloader(train_config)
        self.device = train_config['device']
        
        self.num_train = len(self.train_loader.dataset)
        batch_size = train_config['batch_size']
        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        self.model = PrototypeAD(**model_config).to(self.device)
        
        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        self.model_config = model_config
        self.train_config = train_config
        self.patience = train_config.get('patience', 20)
        self.min_delta = train_config.get('min_delta', 0.005)
        self.writer = train_config.get('writer', None)
        self.run = train_config['run']
        self.dataname = train_config.get('dataset_name', 'unknown')
        self.eval_interval = train_config.get('eval_interval', 10)
        
        print(f"patience={self.patience} with min_delta={self.min_delta}")
        print(f"eval_interval={self.eval_interval}")
        
        self.path = os.path.join(train_config['base_path'], str(train_config['run']))
        os.makedirs(self.path, exist_ok=True)
        
    def get_num_train(self):
        return self.num_train

    def training(self):
        print(self.model_config)
        print(self.train_config)

        self.logger.info(f"Training samples: {self.num_train}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")

        if self.patience is not None:
            best_loss = float('inf')
            patience_cnt = 0
            best_model_state = None
            min_delta = self.min_delta

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_contrastive_loss = 0.0
            running_recon_loss = 0.0
            running_entropy = 0.0
            
            for step, (x_input, y_label, sample_indices) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        output = self.model(x_input, return_dict=True)
                        loss = output['loss']
                        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                            loss = loss.mean()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(x_input, return_dict=True)
                    loss = output['loss']
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                
                contrastive_loss = output['contrastive_loss']
                if isinstance(contrastive_loss, torch.Tensor):
                    contrastive_loss = contrastive_loss.item()
                
                recon_loss = output['reconstruction_loss']
                if isinstance(recon_loss, torch.Tensor):
                    recon_loss = recon_loss.mean().item() if recon_loss.dim() > 0 else recon_loss.item()
                
                entropy = output['entropy']
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.mean().item()

                running_loss += loss.item()
                running_contrastive_loss += contrastive_loss
                running_recon_loss += recon_loss
                running_entropy += entropy
                
            scheduler.step()
            num_batches = len(self.train_loader)
            avg_loss = running_loss / num_batches
            avg_contrastive_loss = running_contrastive_loss / num_batches
            avg_recon_loss = running_recon_loss / num_batches
            avg_entropy = running_entropy / num_batches
            
            info = 'Epoch:[{}]\t loss={:.4f}\t contrastive={:.4f}\t recon={:.4f}\t entropy={:.4f}'
            self.logger.info(info.format(
                epoch, avg_loss, avg_contrastive_loss, avg_recon_loss, avg_entropy
            ))
            
            loss_dict = {
                'total_loss': avg_loss,
                'contrastive_loss': avg_contrastive_loss,
                'reconstruction_loss': avg_recon_loss,
                'entropy': avg_entropy,
            }
            self._log_training(epoch, loss_dict)
            
            if (epoch + 1) % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[Evaluation at Epoch {epoch+1}]")
                metrics = self.evaluate()
                self.logger.info(
                    f"[Epoch {epoch+1}] "
                    f"Recon(RAUC/AP/F1): {metrics['rauc']:.4f}/{metrics['ap']:.4f}/{metrics['f1']:.4f} | "
                    f"Entropy(RAUC/AP/F1): {metrics['entropy_rauc']:.4f}/{metrics['entropy_ap']:.4f}/{metrics['entropy_f1']:.4f}"
                )
                self._log_evaluation(epoch, metrics)
                print(f"{'='*80}\n")
                self.model.train()

            if self.patience is not None:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_cnt = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        print(f"\nEarly Stopping: No Improvement for {self.patience} epochs.")
                        print(f"Best loss: {best_loss:.4f}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        return epoch
        
        print("Training complete.")
        return self.epochs

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate model using reconstruction-based (main) and entropy-based anomaly scores.
        """
        model = self.model
        model.eval()
        
        entropy_scores = []
        recon_scores = []
        test_labels = []
        
        for step, batch in enumerate(self.test_loader):
            if len(batch) == 3:
                x_input, y_label, _ = batch
            else:
                x_input, y_label = batch
                
            x_input = x_input.to(self.device)
            output = model(x_input, return_dict=True)
            
            # Entropy score
            entropy = output['entropy']
            if entropy.dim() > 1:
                entropy = entropy.mean(dim=-1)
            entropy_scores.append(entropy.cpu())
            
            # Reconstruction score
            x_hat = output['x_hat']
            recon = ((x_hat - x_input) ** 2).mean(dim=-1)
            recon_scores.append(recon.cpu())
            
            test_labels.append(y_label)
        
        # Concatenate
        entropy_scores = torch.cat(entropy_scores, dim=0).numpy()
        recon_scores = torch.cat(recon_scores, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        
        # Evaluate with reconstruction (main)
        rauc, ap = aucPerformance(recon_scores, test_labels)
        f1 = F1Performance(recon_scores, test_labels)
        
        # Evaluate with entropy
        entropy_rauc, entropy_ap = aucPerformance(entropy_scores, test_labels)
        entropy_f1 = F1Performance(entropy_scores, test_labels)
        
        # Score statistics
        avg_normal_entropy = entropy_scores[test_labels == 0].mean() if (test_labels == 0).sum() > 0 else 0.0
        avg_abnormal_entropy = entropy_scores[test_labels == 1].mean() if (test_labels == 1).sum() > 0 else 0.0
        avg_normal_recon = recon_scores[test_labels == 0].mean() if (test_labels == 0).sum() > 0 else 0.0
        avg_abnormal_recon = recon_scores[test_labels == 1].mean() if (test_labels == 1).sum() > 0 else 0.0
        
        metric_dict = {
            # Reconstruction-based (main)
            'rauc': float(rauc),
            'ap': float(ap),
            'f1': float(f1),
            # Entropy-based
            'entropy_rauc': float(entropy_rauc),
            'entropy_ap': float(entropy_ap),
            'entropy_f1': float(entropy_f1),
            # Statistics
            'avg_normal_entropy': float(avg_normal_entropy),
            'avg_abnormal_entropy': float(avg_abnormal_entropy),
            'avg_normal_recon': float(avg_normal_recon),
            'avg_abnormal_recon': float(avg_abnormal_recon),
        }
        
        print(f"[Eval] Test samples: {len(test_labels)}")
        print(f"[Eval] Recon   - RAUC: {rauc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
        print(f"[Eval] Entropy - RAUC: {entropy_rauc:.4f}, AP: {entropy_ap:.4f}, F1: {entropy_f1:.4f}")
        
        return metric_dict
    
    def train_test_per_epoch(self, test_per_epochs=10):
        print(self.model_config)
        print(self.train_config)
  
        self.logger.info(f"Training samples: {self.num_train}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        
        metrics = None
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_contrastive_loss = 0.0
            running_recon_loss = 0.0
            running_entropy = 0.0
            
            for step, (x_input, y_label, sample_indices) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x_input, return_dict=True)
                loss = output['loss']
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                    
                contrastive_loss = output['contrastive_loss']
                if isinstance(contrastive_loss, torch.Tensor):
                    contrastive_loss = contrastive_loss.item()
                
                recon_loss = output['reconstruction_loss']
                if isinstance(recon_loss, torch.Tensor):
                    recon_loss = recon_loss.mean().item() if recon_loss.dim() > 0 else recon_loss.item()
                
                entropy = output['entropy']
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.mean().item()

                running_loss += loss.item()
                running_contrastive_loss += contrastive_loss
                running_recon_loss += recon_loss
                running_entropy += entropy
                
                loss.backward()
                optimizer.step()

            scheduler.step()
            num_batches = len(self.train_loader)
            avg_loss = running_loss / num_batches
            avg_contrastive_loss = running_contrastive_loss / num_batches
            avg_recon_loss = running_recon_loss / num_batches
            avg_entropy = running_entropy / num_batches
            
            info = 'Epoch:[{}]\t loss={:.4f}\t contrastive={:.4f}\t recon={:.4f}\t entropy={:.4f}'
            self.logger.info(info.format(
                epoch, avg_loss, avg_contrastive_loss, avg_recon_loss, avg_entropy
            ))
            
            loss_dict = {
                'total_loss': avg_loss,
                'contrastive_loss': avg_contrastive_loss,
                'reconstruction_loss': avg_recon_loss,
                'entropy': avg_entropy,
            }
            self._log_training(epoch, loss_dict)
            
            if (epoch+1) % test_per_epochs == 0:
                metric_dict = self.evaluate()
                
                if metrics is None:
                    metrics = {key: [] for key in metric_dict.keys()}
                
                for key, value in metric_dict.items():
                    metrics[key].append(value)

                print(f"Evaluate on test epoch={epoch+1}")
                self.logger.info(
                    f"[Epoch {epoch+1}] "
                    f"Recon(RAUC/AP/F1): {metric_dict['rauc']:.4f}/{metric_dict['ap']:.4f}/{metric_dict['f1']:.4f} | "
                    f"Entropy(RAUC/AP/F1): {metric_dict['entropy_rauc']:.4f}/{metric_dict['entropy_ap']:.4f}/{metric_dict['entropy_f1']:.4f}"
                )
                
                self._log_evaluation(epoch, metric_dict)
                
                cur_path = os.path.join(self.path, f"model_{epoch+1}.pth")
                torch.save(self.model, cur_path)
                
                self.model.train()

        print("Training complete.")
        return metrics if metrics is not None else {}

    def _log_training(self, epoch, loss_dict):
        """Log training metrics to tensorboard"""
        if self.writer:
            self.writer.add_scalars(f"{self.dataname}/Loss/Total", 
                {f'Run_{self.run}': loss_dict['total_loss']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Loss/Contrastive", 
                {f'Run_{self.run}': loss_dict['contrastive_loss']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Loss/Reconstruction", 
                {f'Run_{self.run}': loss_dict['reconstruction_loss']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Train/Entropy", 
                {f'Run_{self.run}': loss_dict['entropy']}, epoch)
       
            self.writer.flush()

    def _log_evaluation(self, epoch, metrics):
        """Log evaluation metrics to tensorboard"""
        if self.writer:
            # Reconstruction-based (main: rauc, ap, f1)
            self.writer.add_scalars(f"{self.dataname}/Recon/RAUC", 
                {f'Run_{self.run}': metrics['rauc']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Recon/AP", 
                {f'Run_{self.run}': metrics['ap']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Recon/F1", 
                {f'Run_{self.run}': metrics['f1']}, epoch)
            
            # Entropy-based (entropy_rauc, entropy_ap, entropy_f1)
            self.writer.add_scalars(f"{self.dataname}/Entropy/RAUC", 
                {f'Run_{self.run}': metrics['entropy_rauc']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Entropy/AP", 
                {f'Run_{self.run}': metrics['entropy_ap']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Entropy/F1", 
                {f'Run_{self.run}': metrics['entropy_f1']}, epoch)
            
            # Statistics
            self.writer.add_scalars(f"{self.dataname}/Scores/NormalEntropy", 
                {f'Run_{self.run}': metrics['avg_normal_entropy']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Scores/AbnormalEntropy", 
                {f'Run_{self.run}': metrics['avg_abnormal_entropy']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Scores/NormalRecon", 
                {f'Run_{self.run}': metrics['avg_normal_recon']}, epoch)
            self.writer.add_scalars(f"{self.dataname}/Scores/AbnormalRecon", 
                {f'Run_{self.run}': metrics['avg_abnormal_recon']}, epoch)
            
            self.writer.flush()