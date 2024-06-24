import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.utils import inverse_sigmoid
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.torch_utils import model_info, fuse_conv_and_bn
from ultralytics.nn.modules import AIFI, Conv, Concat, RepC3, DWConv, HGStem, HGBlock, RTDETRDecoder, DeformableTransformerDecoder, DeformableTransformerDecoderLayer



class DeformableTransformerDecoder(DeformableTransformerDecoder):

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp, attn_mask=None, padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            bbox = bbox_head[i](output)
            
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox
        return output


class RTDETRDecoder(RTDETRDecoder):

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=6, d_ffn=1024, dropout=0.0, act=nn.ReLU(), eval_idx=-1):
        super().__init__(nc=nc, ch=ch, hd=hd, nq=nq, ndp=ndp, nh=nh, ndl=ndl, d_ffn=d_ffn, dropout=dropout, act=nn.ReLU(), eval_idx=eval_idx)
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = None, None, None, None
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        output = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        return output


class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionAggregator, self).__init__()
        self.feature_dim = feature_dim
        # Query vector for computing attention scores
        self.query = nn.Parameter(torch.Tensor(self.feature_dim))
        nn.init.normal_(self.query)

    def forward(self, x):
        # Compute attention scores
        # [batch_size, seq_length, feature_dim] * [feature_dim, 1] => [batch_size, seq_length, 1]
        attention_scores = torch.matmul(x, self.query.unsqueeze(0).unsqueeze(0).transpose(-1, -2))
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=-1)
        # Compute weighted sum of token embeddings
        # [batch_size, seq_length, 1] * [batch_size, seq_length, feature_dim] => [batch_size, feature_dim]
        aggregated_output = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return aggregated_output


class RTDETR(nn.Module):

    def __init__(self, *args, nc=80, verbose=True):

        super().__init__()

        self.backbone = Backbone()
        self.encoder = EfficientHybridEncoder()
        self.rtdetr_decoder = RTDETRDecoder(nc, [256, 256, 256])
        self.init_weights()

    def init_weights(self, url="https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt", model_path="weights/rtdetr-l.pt"):
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_path):
            # If the file does not exist, download the state dict
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
            # Save the state dict for future use
            torch.save(state_dict, model_path)
            print(f"Downloaded and saved state dict to {model_path}")
        else:
            # Load the state dict from the local file
            state_dict = torch.load(model_path)
            print(f"Loaded state dict from {model_path}")

        loader = WeightLoader()
        loader.load(src=state_dict, tgt=self)

    def extract(self, x):
        s3, s4, s5 = self.backbone(x)
        return s5

    def encode(self, x):
        s3, s4, s5 = self.backbone(x)
        x1, x2, x3 = self.encoder(s3, s4, s5)
        return x3

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, batch=None, **kwargs):
        s3, s4, s5 = self.backbone(x)
        x1, x2, x3 = self.encoder(s3, s4, s5)
        # print(batch["reports"].shape, x3.shape)
        x = self.rtdetr_decoder([x1, x2, x3], batch=batch)
        return x

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]

        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "reports": batch["reports"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )
    
    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

class Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        self.hgstem = HGStem(3, 32, 48)

        self.hgblock_1 = HGBlock(48, 48, 128, 3, 6)
        self.hgblock_2 = HGBlock(128, 96, 512, 3, 6)
        self.hgblock_3 = HGBlock(512, 192, 1024, 5, 6, True, False)
        self.hgblock_4 = HGBlock(1024, 192, 1024, 5, 6, True, True)
        self.hgblock_5 = HGBlock(1024, 192, 1024, 5, 6, True, True)
        self.hgblock_6 = HGBlock(1024, 384, 2048, 5, 6, True, False)

        self.dwconv_1 = DWConv(128, 128, 3, 2, 1, False)
        self.dwconv_2 = DWConv(512, 512, 3, 2, 1, False)
        self.dwconv_3 = DWConv(1024, 1024, 3, 2, 1, False)

    def forward(self, x):
        x = self.hgstem(x)
        x = self.hgblock_1(x)
        x = self.dwconv_1(x)
        x = self.hgblock_2(x) # 3
        s3 = x
        x = self.dwconv_2(x)
        x = self.hgblock_3(x)
        x = self.hgblock_4(x)
        x = self.hgblock_5(x) # 7
        s4 = x
        x = self.dwconv_3(x)
        x = self.hgblock_6(x)
        s5 = x
        return s3, s4, s5



class EfficientHybridEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.aifi = AIFI(256, 1024, 8)
        self.conv = Conv(2048, 256, 1, 1, None, 1, 1, False)
        self.ccfm = CCFM()

    def forward(self, s3, s4, s5):
        x = s5
        x = self.conv(x)
        x = self.aifi(x)
        f5 = x
        x = self.ccfm(s3, s4, f5)
        return x


class Fusion(nn.Module):

    def __init__(self, n, size=None):

        super().__init__()

        self.n = n
        self.concat = Concat(1)
        self.repc = RepC3(512, 256, 3)

        if n == 3:
            assert size is not None
            self.conv_standard = Conv(256, 256, 1, 1)
            self.conv_reduce = Conv(size, 256, 1, 1, None, 1, 1, False)
            self.upsample = nn.Upsample(None, 2, 'nearest')
        if n == 2:
            self.conv = Conv(256, 256, 3, 2)

    def forward(self, a, b): # a: to process b: to concat
        if self.n == 3:
            a = self.conv_standard(a)
            aux = a
            a = self.upsample(a)
            b = self.conv_reduce(b)
            x = self.concat([a, b])
            x = self.repc(x)
            return x, aux
        if self.n == 2:
            a = self.conv(a)
            x = self.concat([a, b])
            x = self.repc(x)
            return x


class CCFM(nn.Module):

    def __init__(self):

        super().__init__()

        self.fusion_1 = Fusion(3, 1024)
        self.fusion_2 = Fusion(3, 512)
        self.fusion_3 = Fusion(2)
        self.fusion_4 = Fusion(2)

    def forward(self, s3, s4, f5):
        x = f5
        x, a = self.fusion_1(x, s4)
        x, b = self.fusion_2(x, s3)
        x1 = x
        x = self.fusion_3(x, b)
        x2 = x
        x = self.fusion_4(x, a)
        x3 = x
        return x1, x2, x3


class CustomModelWeights:

    def __init__(self, custom_model):
        self.aifi = [custom_model.encoder.aifi]
        self.rtdetrdecoder = [custom_model.rtdetr_decoder]

        self.hgstem = [custom_model.backbone.hgstem]
        self.hgblock = [getattr(custom_model.backbone, f"hgblock_{i}") for i in range(1, 6+1)]
        self.dwconv = [getattr(custom_model.backbone, f"dwconv_{i}") for i in range(1, 3+1)]

        fusion_1 = custom_model.encoder.ccfm.fusion_1
        fusion_2 = custom_model.encoder.ccfm.fusion_2
        fusion_3 = custom_model.encoder.ccfm.fusion_3
        fusion_4 = custom_model.encoder.ccfm.fusion_4

        self.conv = [custom_model.encoder.conv, fusion_1.conv_standard, fusion_1.conv_reduce, fusion_2.conv_standard, fusion_2.conv_reduce, fusion_3.conv, fusion_4.conv]
        self.repc3 = [getattr(custom_model.encoder.ccfm, f"fusion_{i}").repc for i in range(1, 4+1)]

class WeightLoader:

    available_tgts = {"custom": ["rtdetr", "custom"]} # tgt -> available srcs
    
    def __init__(self):
        pass

    def fuzzy_load(self, src, tgt):
        src_state_dict = src.state_dict()
        tgt_state_dict = tgt.state_dict()

        try:
            tgt.load_state_dict(src_state_dict)
        except RuntimeError:
            unmatched_weights = []

            for (src_key, src_weight), (tgt_key, tgt_weight) in zip(src_state_dict.items(), tgt_state_dict.items()):
                if src_weight.shape != tgt_weight.shape:
                    unmatched_weights.append(src_key)

            for key in unmatched_weights:
                src_state_dict.pop(key)

            print("WARNING:" + f" shape of {len(unmatched_weights)} weights unmatch when loading weights for {src.__class__.__name__}.")
            tgt.load_state_dict(src_state_dict, strict=False)

    def load(self, src="rtdetr-l.pt", tgt=None):
        assert tgt is not None
        
        if type(src) == str:
            src = torch.load(src)["model"].model
        elif type(src) == dict:
            src = src["model"].model
        tgt_weights = CustomModelWeights(tgt)
        for i in range(len(src)):
            class_name = src[i].__class__.__name__.lower()
            if class_name in ["upsample", "concat"]:
                continue
            src_module = src[i]
            tgt_module = getattr(tgt_weights, class_name).pop(0)

            self.fuzzy_load(tgt_module, src_module)

            




