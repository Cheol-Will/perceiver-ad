# Trainer.py
import os
import random
import torch
import torch.optim as optim
from DataSet.DataLoader import get_multitask_dataloader
from models.LATTEMultitask.Model import LATTEMultitask
from utils import aucPerformance, F1Performance
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader_list, self.test_loader_list, self.num_features_list = get_multitask_dataloader(train_config)
        model_config['num_features_list'] = self.num_features_list
        self.model = LATTEMultitask(**model_config).to(train_config['device'])
        self.epochs = train_config['epochs']
        self.device = train_config['device']
        self.writer = train_config.get('writer', None)
        self.run = train_config['run']  # seed
        self.num_dataset = len(self.train_loader_list)
        self.eval_interval = train_config.get('eval_interval', 10)
        self.checkpoint_interval = train_config.get('checkpoint_interval', 50)
        
        # Use base_path for checkpoint directory (same as summary)
        base_path = train_config.get('base_path', './results')
        self.checkpoint_dir = os.path.join(base_path, 'checkpoints')
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_config.get('lr', 1e-3))
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=train_config.get('scheduler_step', 30),
            gamma=train_config.get('scheduler_gamma', 0.1)
        )
        self.train_config = train_config
        self.model_config = model_config
    def train_test(self):
        """Train and evaluate the model, return metrics at each evaluation epoch"""
        epoch_metrics = {}
        print(f"[Run {self.run}] Starting training for {self.epochs} epochs")
        print(f"Train config: \n{self.train_config}\n")
        print(f"Model config: \n{self.model_config}\n")
        print("-" * 80)
        
        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}/{self.epochs}]")
            
            # Train
            train_loss_list = self.train_one_epoch()
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)
            
            # Log training
            self._log_training(epoch, train_loss_list, avg_train_loss)
            
            # Evaluate periodically
            if (epoch + 1) % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[Evaluation at Epoch {epoch+1}]")
                metric_dict = self.evaluate()
                self._log_evaluation(epoch, metric_dict)
                print(f"{'='*80}\n")
                epoch_metrics[epoch + 1] = metric_dict
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1, avg_train_loss, epoch_metrics.get(epoch + 1))

        return epoch_metrics

    def train_one_epoch(self):
        self.model.train()
        
        running_loss_list = [0.0] * self.num_dataset    
        num_samples_list = [0] * self.num_dataset
        iters = [iter(train_loader) for train_loader in self.train_loader_list]    
        batch_sequence = self.create_batch_sequence()
        
        # Progress bar
        pbar = tqdm(batch_sequence, desc="Training", leave=False)
        
        for dataset_idx in pbar:
            x_input, _ = next(iters[dataset_idx])
            x_input = x_input.to(self.device)
            loss = self.model(x_input, dataset_idx).mean()
            
            running_loss_list[dataset_idx] += loss.item() * x_input.shape[0]
            num_samples_list[dataset_idx] += x_input.shape[0]
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            current_avg_loss = sum(running_loss_list) / max(sum(num_samples_list), 1)
            pbar.set_postfix({'avg_loss': f'{current_avg_loss:.4f}'})
        
        self.scheduler.step()
        
        # Calculate average loss per dataset
        loss_list = [
            running_loss / num_samples if num_samples > 0 else 0.0
            for running_loss, num_samples in zip(running_loss_list, num_samples_list)
        ]
        
        return loss_list
    
    @torch.no_grad()
    def evaluate_one_dataset(self, dataset_idx):
        self.model.eval()
        score, test_label = [], []
        test_loader = self.test_loader_list[dataset_idx]
        
        for x_input, y_label in tqdm(test_loader, desc=f"Eval Dataset {dataset_idx}", leave=False):
            x_input = x_input.to(self.device)
            loss = self.model(x_input, dataset_idx)
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
        return metric_dict

    @torch.no_grad()
    def evaluate(self):
        metric_dict = {}
        for dataset_idx in range(self.num_dataset):
            metric_dict[f"dataset_{dataset_idx}"] = self.evaluate_one_dataset(dataset_idx)
        return metric_dict
    
    def create_batch_sequence(self) -> list[int]:
        batch_sequence = []
        for dataset_idx, loader in enumerate(self.train_loader_list):
            num_batches = len(loader)
            batch_sequence.extend([dataset_idx] * num_batches)
        random.shuffle(batch_sequence)
        return batch_sequence
    
    def _save_checkpoint(self, epoch, train_loss, eval_metrics=None):
        """Save model checkpoint with training state"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"seed_{self.run}_epoch_{epoch}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'seed': self.run,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
        }
        
        # Add evaluation metrics if available
        if eval_metrics is not None:
            checkpoint['eval_metrics'] = eval_metrics
            # Add average metrics for easy access
            avg_rauc = sum(m['rauc'] for m in eval_metrics.values()) / len(eval_metrics)
            avg_ap = sum(m['ap'] for m in eval_metrics.values()) / len(eval_metrics)
            avg_f1 = sum(m['f1'] for m in eval_metrics.values()) / len(eval_metrics)
            checkpoint['avg_metrics'] = {
                'rauc': avg_rauc,
                'ap': avg_ap,
                'f1': avg_f1
            }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  [Checkpoint saved: {checkpoint_path}]")
        
        if eval_metrics is not None:
            print(f"    Average RAUC: {checkpoint['avg_metrics']['rauc']:.4f}")
            print(f"    Average AP:   {checkpoint['avg_metrics']['ap']:.4f}")
            print(f"    Average F1:   {checkpoint['avg_metrics']['f1']:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'avg_metrics' in checkpoint:
            print(f"  Average RAUC: {checkpoint['avg_metrics']['rauc']:.4f}")
            print(f"  Average AP:   {checkpoint['avg_metrics']['ap']:.4f}")
            print(f"  Average F1:   {checkpoint['avg_metrics']['f1']:.4f}")
        
        return checkpoint['epoch']
    
    def _log_training(self, epoch, train_loss_list, avg_train_loss):
        """Log training metrics to console and tensorboard"""
        # Console logging
        print(f"  Average Loss: {avg_train_loss:.4f}")
        for idx, loss in enumerate(train_loss_list):
            print(f"    Dataset {idx}: {loss:.4f}")
        
        # TensorBoard logging
        if self.writer:
            loss_dict = {f"dataset_{idx}": loss for idx, loss in enumerate(train_loss_list)}
            self.writer.add_scalars(f"Run_{self.run}/Train/Loss_per_Dataset", loss_dict, epoch)
            self.writer.add_scalar(f"Run_{self.run}/Train/Average_Loss", avg_train_loss, epoch)
    
    def _log_evaluation(self, epoch, metric_dict):
        """Log evaluation metrics to console and tensorboard"""
        # Console logging
        for dataset_name, metrics in metric_dict.items():
            print(f"\n{dataset_name}:")
            print(f"    RAUC: {metrics['rauc']:.4f}")
            print(f"    AP:   {metrics['ap']:.4f}")
            print(f"    F1:   {metrics['f1']:.4f}")
            print(f"    Avg Normal Score:   {metrics['avg_normal_score']:.4f}")
            print(f"    Avg Abnormal Score: {metrics['avg_abnormal_score']:.4f}")
        
        # TensorBoard logging
        if self.writer:
            for dataset_name, metrics in metric_dict.items():
                self.writer.add_scalars(f"Run_{self.run}/Eval/{dataset_name}/Metrics", 
                                       {
                                           'RAUC': metrics['rauc'],
                                           'AP': metrics['ap'],
                                           'F1': metrics['f1']
                                       }, epoch)
                
                self.writer.add_scalars(f"Run_{self.run}/Eval/{dataset_name}/Scores", 
                                       {
                                           'Avg_Normal_Score': metrics['avg_normal_score'],
                                           'Avg_Abnormal_Score': metrics['avg_abnormal_score']
                                       }, epoch)
            
            # Aggregate metrics across datasets
            avg_rauc = sum(m['rauc'] for m in metric_dict.values()) / len(metric_dict)
            avg_ap = sum(m['ap'] for m in metric_dict.values()) / len(metric_dict)
            avg_f1 = sum(m['f1'] for m in metric_dict.values()) / len(metric_dict)
            
            self.writer.add_scalars(f"Run_{self.run}/Eval/Average/Metrics", 
                                   {
                                       'RAUC': avg_rauc,
                                       'AP': avg_ap,
                                       'F1': avg_f1
                                   }, epoch)