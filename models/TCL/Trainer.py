import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from DataSet.DataLoader import get_dataloader
from models.TCL.Model import TCL
from utils import aucPerformance, F1Performance
import copy

class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        self.device = train_config['device']
        self.model = TCL(**model_config).to(self.device)

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
                        output = self.model(x_input)
                        loss = output['loss'].mean()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(x_input)
                    loss = output['loss'].mean()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                optimizer.zero_grad()

            scheduler.step()
            avg_loss = running_loss / len(self.train_loader)

            # Log training
            info = 'Epoch:[{}]\t loss={:.6f}'
            self.logger.info(info.format(epoch, avg_loss))
            self._log_training(epoch, avg_loss)
            # Evaluate periodically
            if (epoch + 1) % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[Evaluation at Epoch {epoch+1}]")
                metrics = self.evaluate()
                score_configs = [('Contra', '')]
                for name, prefix in score_configs:
                    self.logger.info(
                        f"[Epoch {epoch+1}] {name:8s} | "
                        f"AUC-ROC: {metrics[f'{prefix}rauc']:.4f} | "
                        f"AUC-PR: {metrics[f'{prefix}ap']:.4f} | "
                        f"F1: {metrics[f'{prefix}f1']:.4f}"
                    )
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
                        return epoch
        print("Training complete.")
        return self.epochs            

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        print("Build memory bank for evaluation")
        model.build_eval_memory_bank(self.train_loader, self.device, False) # do not use amp during inference
        
        score_list, test_label_list = [], []

        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            output = model(x_input)
            
            contra = output['contrastive_score'].data.cpu()
            
            score_list.append(contra)
            test_label_list.append(y_label)
        
        model.empty_eval_memory_bank() # empty memory
        model.train()

        score = torch.cat(score_list, axis=0).numpy()
        test_label = torch.cat(test_label_list, axis=0).numpy()
        
        def calc_metrics(scores, labels, prefix=''):
            rauc, ap = aucPerformance(scores, labels)
            f1 = F1Performance(scores, labels)
            avg_normal = scores[labels == 0].mean()
            avg_abnormal = scores[labels == 1].mean()
            
            return {
                f'{prefix}rauc': float(rauc),
                f'{prefix}ap': float(ap),
                f'{prefix}f1': float(f1),
                f'{prefix}avg_normal': float(avg_normal),
                f'{prefix}avg_abnormal': float(avg_abnormal)
            }

        metric_dict = calc_metrics(score, test_label, prefix='') # recon

        return metric_dict
    
    def _log_training(self, epoch, avg_loss):
        """Log all training losses to tensorboard"""
        if self.writer:
            self.writer.add_scalars(f"{self.dataname}/Loss/Total", 
                {f'Run_{self.run}': avg_loss}, epoch)
            
            self.writer.flush()
    def _log_evaluation(self, epoch, metrics):
        if self.writer:
            score_types = {
                'Contra': '',
            }
            
            for name, prefix in score_types.items():
                self.writer.add_scalars(f"{self.dataname}/Metrics/{name}_RAUC", 
                    {f'Run_{self.run}': metrics[f'{prefix}rauc']}, epoch)
                self.writer.add_scalars(f"{self.dataname}/Metrics/{name}_AP", 
                    {f'Run_{self.run}': metrics[f'{prefix}ap']}, epoch)
                self.writer.add_scalars(f"{self.dataname}/Metrics/{name}_F1", 
                    {f'Run_{self.run}': metrics[f'{prefix}f1']}, epoch)
                self.writer.add_scalars(f"{self.dataname}/Scores/{name}_Normal_Avg", 
                    {f'Run_{self.run}': metrics[f'{prefix}avg_normal']}, epoch)
                self.writer.add_scalars(f"{self.dataname}/Scores/{name}_Abnormal_Avg", 
                    {f'Run_{self.run}': metrics[f'{prefix}avg_abnormal']}, epoch)
            
            self.writer.flush()