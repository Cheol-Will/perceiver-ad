import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from DataSet.DataLoader import get_dataloader
from models.TDACL.Model import TDACL
from utils import aucPerformance, F1Performance
import copy


class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        self.device = train_config['device']
        self.model = TDACL(**model_config).to(self.device)

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

    def _fill_metrics_per_epoch(self, eval_records, last_metrics):
        planned = list(range(self.eval_interval, self.epochs + 1, self.eval_interval))
        if not planned or planned[-1] != self.epochs:
            planned.append(self.epochs)

        if last_metrics is None:
            last_metrics = self.evaluate()

        flat = {}
        for ep in planned:
            if ep in eval_records:
                last_metrics = eval_records[ep]
            for k, v in last_metrics.items():
                if isinstance(v, (int, float)):
                    flat[f"ep{ep}_{k}"] = float(v)
        return flat

    def training(self):
        print(self.model_config)
        print(self.train_config)

        self.logger.info(self.train_loader.dataset.data[0])
        self.logger.info(self.test_loader.dataset.data[0])

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")

        eval_records = {}
        last_eval_metrics = None

        if self.patience is not None:
            best_loss = float('inf')
            patience_cnt = 0
            best_model_state = None
            min_delta = self.min_delta

        for epoch in range(self.epochs):
            running_loss = 0.0

            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast():
                        output = self.model(x_input)
                        loss = output['loss']

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(x_input)
                    loss = output['loss']

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            scheduler.step()

            avg_loss = running_loss / len(self.train_loader)
            info = 'Epoch:[{}]\t loss={:.6f}'
            self.logger.info(info.format(epoch, avg_loss))
            self._log_training(epoch, avg_loss)

            if (epoch + 1) % self.eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"[Evaluation at Epoch {epoch+1}]")
                metrics = self.evaluate()
                eval_records[epoch + 1] = metrics
                last_eval_metrics = metrics

                score_configs = [('KNN', ''),]
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
                            path = os.path.join(self.path, "model.pth")
                            torch.save(self.model, path)
                        metrics_per_epoch = self._fill_metrics_per_epoch(eval_records, last_eval_metrics)
                        return {
                            "epochs_ran": int(epoch + 1),
                            "early_stopped": True,
                            "best_loss": float(best_loss),
                            "metrics_per_epoch": metrics_per_epoch,
                        }

        print("Training complete.")
        path = os.path.join(self.path, "model.pth")
        torch.save(self.model, path)

        metrics_per_epoch = self._fill_metrics_per_epoch(eval_records, last_eval_metrics)
        return {
            "epochs_ran": int(self.epochs),
            "early_stopped": False,
            "metrics_per_epoch": metrics_per_epoch,
        }

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        print("Build memory bank for evaluation")
        model.build_eval_memory_bank(self.train_loader, self.device, False)

        score_list, test_label_list = [], []
        contra_score_list, combined_score_list = [], []
        
        def _to_safe_cpu_1d(x: torch.Tensor):
            x = x.detach()
            if x.dim() > 1:
                x = x.view(-1)
            x = x.to(torch.float64).cpu()
            x = torch.nan_to_num(x, nan=0.0, posinf=1e12, neginf=-1e12)
            x = torch.clamp(x, -1e12, 1e12)

            return x

        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            output = model(x_input)

            score = _to_safe_cpu_1d(output['score'])
            score_list.append(score)
            test_label_list.append(y_label)

        score = torch.cat(score_list, axis=0).numpy()
        test_label = torch.cat(test_label_list, axis=0).numpy()

        model.empty_eval_memory_bank()
        model.train()


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

        metric_dict = calc_metrics(score, test_label, prefix='')
        return metric_dict

    def _log_training(self, epoch, avg_loss):
        if self.writer:
            self.writer.add_scalars(f"{self.dataname}/Loss/Total", {f'Run_{self.run}': avg_loss}, epoch)
            self.writer.flush()

    def _log_evaluation(self, epoch, metrics):
        if self.writer:
            score_types = {'KNN': ''}
            for name, prefix in score_types.items():
                self.writer.add_scalars(
                    f"{self.dataname}/Metrics/{name}_AP",
                    {f'Run_{self.run}': metrics[f'{prefix}ap']},
                    epoch
                )
                self.writer.add_scalars(
                    f"{self.dataname}/Scores/{name}_Normal_Avg",
                    {f'Run_{self.run}': metrics[f'{prefix}avg_normal']},
                    epoch
                )
                self.writer.add_scalars(
                    f"{self.dataname}/Scores/{name}_Abnormal_Avg",
                    {f'Run_{self.run}': metrics[f'{prefix}avg_abnormal']},
                    epoch
                )
            self.writer.flush()