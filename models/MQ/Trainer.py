import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from DataSet.DataLoader import get_mq_dataloader
from models.MQ.Model import MQ
from utils import aucPerformance, F1Performance
import math
import copy


class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        # Get MQ-specific dataloaders
        self.train_loader, self.test_loader = get_mq_dataloader(train_config)
        self.device = train_config['device']
        
        # Get number of training samples
        self.num_train = len(self.train_loader.dataset)
        batch_size = train_config['batch_size']
        
        # Compute optimal queue size
        original_queue_size = model_config.get('queue_size', 1024)
        actual_queue_size = min(original_queue_size, self.num_train)

        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        print(f"="*60)
        print(f"MQ Model Configuration:")
        print(f"  - Number of training samples: {self.num_train}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Original queue size: {original_queue_size}")
        print(f"  - Actual queue size: {actual_queue_size}")
        print(f"  - Use AMP: {self.use_amp}")
        print(f"="*60)
        
        # Update model config with actual queue size and num_train
        model_config['queue_size'] = actual_queue_size
        
        self.model = MQ(**model_config).to(self.device)
        
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
            running_recon_loss = 0.0
            
            for step, (x_input, y_label, sample_indices) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                optimizer.zero_grad()
                
                # AMP forward pass
                if self.use_amp:
                    with autocast():
                        output = self.model(x_input, return_dict=True)
                        loss = output['loss'].mean()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(x_input, return_dict=True)
                    loss = output['loss'].mean()
                    loss.backward()
                    optimizer.step()
                
                recon_loss = output['reconstruction_loss'].mean()

                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                
            scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            avg_recon_loss = running_recon_loss / len(self.train_loader)
            
            info = 'Epoch:[{}]\t loss={:.4f}\t recon={:.4f}'
            self.logger.info(info.format(
                epoch, avg_loss, avg_recon_loss
            ))
            
            loss_dict = {
                'total_loss': avg_loss,
                'reconstruction_loss': avg_recon_loss,
            }
            self._log_training(epoch, loss_dict)
            
            # Evaluate periodically with FULL memory bank
            if (epoch + 1) % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[Evaluation at Epoch {epoch+1}]")
                metrics = self.evaluate()
                self.logger.info(
                    f"[Epoch {epoch+1}] AUC-ROC: {metrics['rauc']:.4f} | "
                    f"AUC-PR: {metrics['ap']:.4f} | F1: {metrics['f1']:.4f}"
                )
                self._log_evaluation(epoch, metrics)
                print(f"{'='*80}\n")
                self.model.train()  # Resume training mode

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
        Evaluate model using full memory bank built from all training samples.
        """
        model = self.model
        model.eval()
        
        print("[Eval] Building full memory bank...")
        model.build_eval_memory_bank(self.train_loader, self.device)
        
        score, test_label = [], []
        
        for step, batch in enumerate(self.test_loader):
            # Handle both (x, y, idx) and (x, y) formats
            if len(batch) == 3:
                x_input, y_label, _ = batch
            else:
                x_input, y_label = batch
                
            x_input = x_input.to(self.device)
            output = model(x_input, return_dict=True, use_eval_memory=True)
            anomaly_score = output['anomaly_score']
            anomaly_score = anomaly_score.data.cpu()
            score.append(anomaly_score)
            test_label.append(y_label)
            
        score = torch.cat(score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        
        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        
        avg_normal_score = score[test_label == 0].mean()
        avg_abnormal_score = score[test_label == 1].mean()
        
        model.clear_eval_memory_bank()
        
        metric_dict = {
            'rauc': float(rauc),
            'ap': float(ap),
            'f1': float(f1),
            'avg_normal_score': float(avg_normal_score), 
            'avg_abnormal_score': float(avg_abnormal_score)
        } 
        
        print(f"[Eval] Memory bank size used: {self.num_train}")
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
        
        metrics = {
            'rauc': [],
            'ap': [],
            'f1': [],
            'avg_normal_score': [],
            'avg_abnormal_score': []
        }
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_recon_loss = 0.0
            
            for step, (x_input, y_label, sample_indices) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                
                output = self.model(x_input, return_dict=True)
                loss = output['loss'].mean()
                recon_loss = output['reconstruction_loss'].mean()

                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            avg_recon_loss = running_recon_loss / len(self.train_loader)
            
            
            info = 'Epoch:[{}]\t loss={:.4f}\t recon={:.4f}\t'
            self.logger.info(info.format(
                epoch, avg_loss, avg_recon_loss
            ))
            
            loss_dict = {
                'total_loss': avg_loss,
                'reconstruction_loss': avg_recon_loss,
            }
            self._log_training(epoch, loss_dict)
            
            if (epoch+1) % test_per_epochs == 0:
                # Use full memory bank for evaluation
                metric_dict = self.evaluate()
                metrics['rauc'].append(metric_dict['rauc'])
                metrics['ap'].append(metric_dict['ap'])
                metrics['f1'].append(metric_dict['f1'])
                metrics['avg_normal_score'].append(metric_dict['avg_normal_score'])
                metrics['avg_abnormal_score'].append(metric_dict['avg_abnormal_score'])

                print(f"Evaluate on test epoch={epoch+1}")
                self.logger.info(
                    f"[Epoch {epoch+1}] AUC-ROC: {metric_dict['rauc']:.4f} | "
                    f"AUC-PR: {metric_dict['ap']:.4f} | F1: {metric_dict['f1']:.4f}"
                )
                
                self._log_evaluation(epoch, metric_dict)
                
                cur_path = os.path.join(self.path, f"model_{epoch+1}.pth")
                torch.save(self.model, cur_path)
                
                self.model.train()  # Resume training mode

        print("Training complete.")
        return metrics

    def _log_training(self, epoch, loss_dict):
        """Log training metrics to tensorboard"""
        if self.writer:
            self.writer.add_scalars(f"{self.dataname}/Loss/Total", 
                {f'Run_{self.run}': loss_dict['total_loss']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Loss/Reconstruction", 
                {f'Run_{self.run}': loss_dict['reconstruction_loss']}, epoch)
       
            self.writer.flush()

    def _log_evaluation(self, epoch, metrics):
        """Log evaluation metrics to tensorboard"""
        if self.writer:
            self.writer.add_scalars(f"{self.dataname}/Metrics/RAUC", 
                {f'Run_{self.run}': metrics['rauc']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Metrics/AP", 
                {f'Run_{self.run}': metrics['ap']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Metrics/F1", 
                {f'Run_{self.run}': metrics['f1']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Scores/Normal", 
                {f'Run_{self.run}': metrics['avg_normal_score']}, epoch)
            
            self.writer.add_scalars(f"{self.dataname}/Scores/Abnormal", 
                {f'Run_{self.run}': metrics['avg_abnormal_score']}, epoch)
            
            self.writer.flush()