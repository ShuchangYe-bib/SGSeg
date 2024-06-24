import torch

from torchmetrics import Metric
from nltk.translate import meteor_score
from torchmetrics.text import BLEUScore, ROUGEScore

class BLEUScore(BLEUScore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            super().update([pred], [[target]])
            
class ROUGEScore(ROUGEScore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, rouge_keys="rougeL")

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            super().update(pred, target)
    
    def compute(self):
        rouge_scores = super().compute()
        return rouge_scores['rougeL_fmeasure'] if rouge_scores else None

class METEORScore(Metric):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_state("meteor", default=[], dist_reduce_fx=None)

    def update(self, preds, targets):
        preds = [[self.tokenizer._convert_id_to_token(x) for x in self.tokenizer.encode(text)] for text in preds]
        targets = [[self.tokenizer._convert_id_to_token(x) for x in self.tokenizer.encode(text)] for text in targets]
        
        meteor_scores = [meteor_score.single_meteor_score(t, p) for t, p in zip(targets, preds)]
        self.meteor.extend(meteor_scores)

    def compute(self):
        if len(self.meteor) == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.meteor).mean()