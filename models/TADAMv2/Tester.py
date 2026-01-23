import os
import torch
from DataSet.DataLoader import get_dataloader
from models.TADAM.Trainer import Trainer
from utils import aucPerformance, F1Performance

class Tester(Trainer):
    def __init__(self, model_config: dict, train_config: dict):
        super().__init__(model_config, train_config)
        self.pth_path = os.path.join(train_config['base_pth_path'], str(train_config['run']))
        path = os.path.join(self.pth_path, "model.pth")
        assert os.path.exists(path)
        print(f"Load weights from {path}")
        self.model.load_state_dict(torch.load(path))

    def evaluate(self):
        model = self.model
        model.eval()
        print("Build attention bank for evaluation")
        model.build_eval_attn_bank(self.train_loader, self.device, False)
        recon_list, test_label_list = [], []
        combined_score_dict = {}
        cls_knn_lists = {top_k: [] for top_k in [1, 5, 10, 16, 32, 64]}
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)

            output = model.forward_knn_cls(x_input)
            cls_scores = output['cls_scores']
            for top_k in cls_knn_lists.keys():
                cls_knn_lists[top_k].append(cls_scores[f'cls_knn{top_k}'].detach().cpu())

            # recon + lambda * knn_score
            output = model.forward_combined(x_input, use_cls=False)['combined']
            for k, v in output.items():
                v = v.detach().cpu()
                if k in combined_score_dict:
                    combined_score_dict[k].append(v)
                else:
                    combined_score_dict[k] = [v]

            # recon + lambda * cls_knn_score
            output = model.forward_combined(x_input, use_cls=True)['combined']
            for k, v in output.items():
                v = v.detach().cpu()
                if k in combined_score_dict:
                    combined_score_dict[k].append(v)
                else:
                    combined_score_dict[k] = [v]

            test_label_list.append(y_label)

        model.empty_eval_attn_bank()
        model.train()

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
                f'{prefix}avg_normal_score': float(avg_normal),
                f'{prefix}avg_abnormal_score': float(avg_abnormal),
            }
        
        cls_knn_score = torch.cat(cls_knn_lists[1], axis=0).numpy()
        metric_dict = calc_metrics(cls_knn_score, test_label, prefix='')

        for k in [1, 5, 10, 16, 32, 64]:
            knn_score = torch.cat(cls_knn_lists[k], axis=0).numpy()
            metric_dict.update(calc_metrics(knn_score, test_label, prefix=f'cls_knn{k}_'))

        for k, v in combined_score_dict.items():
            combined_score = torch.cat(v, axis=0).numpy()
            metric_dict.update(calc_metrics(combined_score, test_label, prefix=f'{k}_'))


        return metric_dict