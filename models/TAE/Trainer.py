import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from DataSet.DataLoader import get_dataloader
from models.TAE.Model import TAE
from utils import aucPerformance, F1Performance
import copy

class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        self.device = train_config['device']        
        
        self.model = TAE(
            **model_config
        ).to(self.device)

        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        self.model_config = model_config
        self.train_config = train_config
        self.patience = train_config['patience']
        self.min_delta = train_config['min_delta']
        self.writer = train_config.get('writer', None)
        self.run = train_config['run']
        self.dataname = train_config.get('dataset_name', 'unknown')
        self.eval_interval = train_config.get('eval_interval', 10)
        
        print(f"patience={self.patience} with min_delta={self.min_delta}")
        print(f"eval_interval={self.eval_interval}")
        self.path = os.path.join(train_config['base_path'], str(train_config['run']))
        os.makedirs(self.path, exist_ok=True)
        
    def training(self):
        print(self.model_config)
        print(self.train_config)

        self.logger.info(self.train_loader.dataset.data[0])
        self.logger.info(self.test_loader.dataset.data[0])

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
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)

                if self.use_amp:
                    with autocast():
                        loss = self.model(x_input).mean()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss = self.model(x_input).mean()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                optimizer.zero_grad()

            scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            
            # Log training
            info = 'Epoch:[{}]\t loss={:.6f}\t'
            self.logger.info(info.format(epoch, avg_loss))
            self._log_training(epoch, avg_loss)
            
            # Evaluate periodically
            if (epoch + 1) % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[Evaluation at Epoch {epoch+1}]")
                metrics = self.evaluate()
                self.logger.info(f"[Epoch {epoch+1}] AUC-ROC: {metrics['rauc']:.4f} | AUC-PR: {metrics['ap']:.4f} | F1: {metrics['f1']:.4f}")
                self._log_evaluation(epoch, metrics)
                print(f"{'='*80}\n")

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
                            path = os.path.join(self.path, "model.pth")
                            torch.save(self.model, path)
                        return epoch
        
        print("Training complete.")
        path = os.path.join(self.path, "model.pth")
        torch.save(self.model, path)

        return self.epochs

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            loss = model(x_input)
            loss = loss.data.cpu()
            score.append(loss)
            test_label.append(y_label)
        score = torch.cat(score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        avg_normal_score = score[test_label == 0].mean()
        avg_abnormal_score = score[test_label == 1].mean()
        
        metric_dict = {
            'rauc': float(rauc),
            'ap': float(ap),
            'f1': float(f1),
            'avg_normal_score': float(avg_normal_score), 
            'avg_abnormal_score': float(avg_abnormal_score)
        } 
        self.model.train()
        return metric_dict
    
    def train_test_per_epoch(self, test_per_epochs = 50):
        print(self.model_config)
        print(self.train_config)
  
        self.logger.info(self.train_loader.dataset.data[0])
        self.logger.info(self.test_loader.dataset.data[0])

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
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                loss = self.model(x_input).mean()

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            
            # Log training
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            self.logger.info(info.format(epoch, avg_loss))
            self._log_training(epoch, avg_loss)
            
            if (epoch+1) % test_per_epochs == 0:
                metric_dict = self.evaluate()
                metrics['rauc'].append(metric_dict['rauc'])
                metrics['ap'].append(metric_dict['ap'])
                metrics['f1'].append(metric_dict['f1'])
                metrics['avg_normal_score'].append(metric_dict['avg_normal_score'])
                metrics['avg_abnormal_score'].append(metric_dict['avg_abnormal_score'])

                print(f"Evaluate on test epoch={epoch+1}")
                self.logger.info(f"[Epoch {epoch+1}] AUC-ROC: {metric_dict['rauc']:.4f} | AUC-PR: {metric_dict['ap']:.4f} | F1: {metric_dict['f1']:.4f}")
                
                # Log evaluation
                self._log_evaluation(epoch, metric_dict)
                
                cur_path = os.path.join(self.path, f"model_{epoch+1}.pth")
                torch.save(self.model, cur_path)

        print("Training complete.")
        return metrics
    
    def _log_training(self, epoch, avg_loss):
        """Log training metrics to tensorboard (MQ style)"""
        if self.writer:
            self.writer.add_scalars(f"{self.dataname}/Loss/Total", 
                {f'Run_{self.run}': avg_loss}, epoch)
            
            self.writer.flush()
    
    def _log_evaluation(self, epoch, metrics):
        """Log evaluation metrics to tensorboard (MQ style)"""
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