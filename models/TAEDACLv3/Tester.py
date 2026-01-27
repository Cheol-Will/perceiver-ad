import os
import torch
from models.TAEDACLv3.Trainer import Trainer
from utils import aucPerformance, F1Performance
from collections import defaultdict

class Tester(Trainer):
    def __init__(self, model_config: dict, train_config: dict):
        super().__init__(model_config, train_config)
        self.pth_path = os.path.join(train_config['base_pth_path'], str(train_config['run']))
        path = os.path.join(self.pth_path, "model.pth")
        assert os.path.exists(path)
        print(f"Load weights from {path}")
        obj = torch.load(path, map_location=self.device)
        state = obj if isinstance(obj, dict) else obj.state_dict()
        self.model.load_state_dict(state)


    def evaluate(self):
        model = self.model
        model.eval()
        print("Build attention bank for evaluation")
        model.build_eval_memory_bank(self.train_loader, self.device, False)
        model.build_eval_attn_bank(self.train_loader, self.device, False)

        weight_list = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
        keyword_list = [
            'knn', 
            'knn_attn_cls', 
            'knn_attn', 
            'knn_attn_penul', 
            'knn_attn_cls_penul',
            'knn_attn_first', 
            'knn_attn_cls_first',
        ]

        recon_list, test_label_list = [], []
        combined_score_dict = defaultdict(list)

        def _append_combined(d, combined):
            for k, v in combined.items():
                d[k].append(v.detach().cpu())

        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)

            output = model(x_input)
            recon_list.append(output['recon_loss'].detach().cpu())

            for keyword in keyword_list:
                output = model.forward_combined(x_input, keyword, weight_list)
                _append_combined(combined_score_dict, output['combined'])
                _append_combined(combined_score_dict, output['knn_scores'])

            test_label_list.append(y_label)

        model.empty_eval_attn_bank()
        model.train()

        recon_score = torch.cat(recon_list, axis=0).numpy()
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

        metric_dict = calc_metrics(recon_score, test_label, prefix='')

        for k, v in combined_score_dict.items():
            combined_score = torch.cat(v, axis=0).numpy()
            metric_dict.update(calc_metrics(combined_score, test_label, prefix=f'{k}_'))

        return metric_dict
