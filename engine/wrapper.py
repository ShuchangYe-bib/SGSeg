import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from copy import deepcopy
from tabulate import tabulate
from monai.losses import DiceCELoss
from torch.autograd import Variable
from utils.model import SGSeg
from torchmetrics import Accuracy, Dice
from utils.metrics import BLEUScore, ROUGEScore, METEORScore
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy, BinaryRecall, BinaryPrecision, MultilabelF1Score

class SGSegWrapper(pl.LightningModule):

    def __init__(self, args, inference=False):
        super(SGSegWrapper, self).__init__()
        
        self.inference = inference
        self.model = SGSeg(args.bert_type, args.vision_type, args.project_dim)
        
        if inference:
            return
        
        self.lr = args.lr
        self.history = {}
        self.switch = True
        self.clip_epoch = 20
        self.running_epoch = 0
        
        # Loss functions
        self.loss_seg = DiceCELoss()
        self.loss_class = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_contrastive = ContrastiveLoss()
        
        # Metrics for segmentation, detection, and text generation
        seg_metrics = nn.ModuleDict({
            "seg_acc": Accuracy(task='binary'), 
            "dice": Dice(), 
            "MIoU": BinaryJaccardIndex()
        })
        detect_metrics = nn.ModuleDict({
            "acc": BinaryAccuracy(), 
            "recall": BinaryRecall(), 
            "precision": BinaryPrecision(), 
            "f1_score": MultilabelF1Score(num_labels=args.num_labels)
        })
        text_metrics = nn.ModuleDict({
            "bleu_1": BLEUScore(n_gram=1), 
            "bleu_2": BLEUScore(n_gram=2), 
            "bleu_3": BLEUScore(n_gram=3), 
            "bleu_4": BLEUScore(n_gram=4), 
            "rouge": ROUGEScore(), 
            "meteor": METEORScore(tokenizer=self.model.tokenizer)
        })
        
        self.train_metrics = {"seg": seg_metrics, "detect": detect_metrics, "text": text_metrics}
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()
        
        # Setting up metric columns for records
        metrics = ["loss", *seg_metrics.keys(), *detect_metrics.keys(), *text_metrics.keys()]
        columns = [f"{stage}_" + metric for stage in ["train", "val", "test"] for metric in metrics]
        self.seg_columns = [f"{stage}_" + metric for stage in ["train", "val", "test"] for metric in ["loss"] + list(seg_metrics.keys())]
        self.detect_columns = [f"{stage}_" + metric for stage in ["train", "val", "test"] for metric in detect_metrics.keys()]
        self.text_columns = [f"{stage}_" + metric for stage in ["train", "val", "test"] for metric in text_metrics.keys()]
        
        self.records = pd.DataFrame(columns=columns)
        self.records.index.name = "epoch"
        self.records_path = None

    def configure_optimizers(self):
        # Configure optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        
    def forward(self, x, image_size=(224, 224)):
        return self.model.forward(x, inference=self.inference, image_size=image_size)

    def gen(self, x, **kwargs):
        return self.model.gen(x, **kwargs)

    def shared_step(self, batch, batch_idx, mode):
        x, y = batch
        image, text, label = x
        if mode == "train":
            preds = self.model.forward([image, text])
            pred_text = self.model.gen(image)
            pred_seg, pred_detect = preds
            if self.running_epoch < self.clip_epoch:
                features = self.model.features(image, text)
                loss = self.loss_contrastive(*features)
            else:
                loss = self.loss_seg(pred_seg, y) + self.loss_bce(pred_detect, label)
            self.running_epoch += 1
        else: 
            pred_text = self.model.gen(image)
            preds = self.model.forward([image, pred_text])
            pred_seg, pred_detect = preds
            loss = self.loss_seg(pred_seg, y) + self.loss_bce(pred_detect, label)
        return {
            'loss': loss, 
            'pred_seg': pred_seg.detach().cpu(), 
            'y_seg': y.detach().cpu(),
            'pred_detect': pred_detect.detach().cpu(),
            'y_detect': label.detach().cpu(),
            'pred_text': pred_text,
            'y_text': text
        }
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "valid")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self, outputs, stage):
        metrics = getattr(self, f"{stage}_metrics")
        for metrics_type, type_metrics in metrics.items():
            for name, metric in type_metrics.items():
                step_metric = metric(outputs[f'pred_{metrics_type}'], outputs[f'y_{metrics_type}']).item()
                if stage == "train":
                    self.log(name, step_metric, prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss': self.shared_step_end(outputs, "train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss': self.shared_step_end(outputs, "val")}
            
    def test_step_end(self, outputs):
        return {'test_loss': self.shared_step_end(outputs, "test")}
            
    def shared_epoch_end(self, outputs, stage="train"):
        metrics = getattr(self, f"{stage}_metrics")
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage + "_loss").replace('train_', '')] for t in outputs])).item()
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for metrics_type, type_metrics in metrics.items():
            for name, metric in type_metrics.items():
                epoch_metric = metric.compute().item()
                metric.reset()
                dic[stage + "_" + name] = epoch_metric
        dic["monitor_metric"] = dic[stage + "_seg_acc"] + dic[stage + "_dice"]
        
        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print_metrics(dic, stage="train")
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_metrics(dic, stage="val")
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)
        
        # Log when reaching the best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            self.print("<<<<<< reach best {0} : {1} >>>>>>\n".format(monitor, arr_scores[best_score_idx]), file=sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        dic.pop("epoch", None)
        self.print_metrics(dic, stage="test")
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_metrics(self, metrics, stage):
        print()
        if self.records_path is None:
            model_path = self.trainer.checkpoint_callback._get_metric_interpolated_filepath_name(metrics, self.trainer)
            model_dir = os.path.dirname(model_path)
            self.records_path = os.path.splitext(model_path)[0] + ".csv"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        data = pd.DataFrame(metrics, index=[0], columns=metrics.keys())
        to_drop = set(data.columns) - set(self.records.columns)
        data = data.drop(to_drop, axis=1)

        if "epoch" in metrics:
            epoch = int(metrics["epoch"])
            self.records.loc[epoch, data.columns] = data.values
        else:
            self.records.loc[0, data.columns] = data.values

        if stage == "train":
            data.dropna(axis=1).to_csv(self.records_path)
        data = self.records.tail(1)

        print("\nSegmentation metrics")
        seg_data = data[self.seg_columns]
        seg_data = seg_data.values.T.reshape(-1, len(self.seg_columns) // 3)
        seg_data = pd.DataFrame(seg_data, columns=["loss"] + list(self.train_metrics["seg"].keys()), index=["train", "val", "test"])
        print(tabulate(seg_data.dropna(axis=0), headers="keys", tablefmt='psql', floatfmt=".4f"))

        print("\nDetection metrics")
        detect_data = data[self.detect_columns]
        detect_data = detect_data.values.T.reshape(-1, len(self.detect_columns) // 3)
        detect_data = pd.DataFrame(detect_data, columns=self.train_metrics["detect"].keys(), index=["train", "val", "test"])
        print(tabulate(detect_data.dropna(axis=0), headers="keys", tablefmt='psql', floatfmt=".4f"))

        print("\nText generation metrics")
        text_data = data[self.text_columns]
        text_data = text_data.values.T.reshape(-1, len(self.text_columns) // 3)
        text_data = pd.DataFrame(text_data, columns=self.train_metrics["text"].keys(), index=["train", "val", "test"])
        print(tabulate(text_data.dropna(axis=0), headers="keys", tablefmt='psql', floatfmt=".4f"))

        print()

class ContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, cnn_code, rnn_code):
        return self.global_loss(cnn_code, rnn_code)

    def global_loss(self, cnn_code, rnn_code, eps=1e-8, temp3=10.0):

        if cnn_code.dim() == 3:
            cnn_code = cnn_code.mean(dim=1)
        if rnn_code.dim() == 3:
            rnn_code = rnn_code.mean(dim=1)

        batch_size = cnn_code.shape[0]
        labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * temp3

        scores0 = scores0.squeeze()

        scores1 = scores0.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        return (loss0 + loss1) / 2


