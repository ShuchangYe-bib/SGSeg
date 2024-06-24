import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from monai.transforms import Compose, ToTensord
from transformers import AutoTokenizer, AutoModel

from .rtdetr import RTDETR, AttentionAggregator

class BERTModel(nn.Module):
    """
    BERT-based model for extracting text features.
    """

    def __init__(self, bert_type, project_dim):
        """
        Initialize the BERTModel.

        Args:
            bert_type (str): Type of BERT model.
            project_dim (int): Dimension of the projection layer.
        """
        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        # Set model parameters to require gradient computation
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the BERTModel.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            dict: Dictionary containing hidden states and projected embeddings.
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        # Combine first, second, and last hidden states
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]])  # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  # pooling
        embed = self.project_head(embed)

        return {'feature': output['hidden_states'], 'project': embed}

class VisionModel(nn.Module):
    """
    Vision model for extracting image features.
    """

    def __init__(self, vision_type, project_dim):
        """
        Initialize the VisionModel.

        Args:
            vision_type (str): Type of vision model.
            project_dim (int): Dimension of the projection layer.
        """
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True)
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        """
        Forward pass for the VisionModel.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            dict: Dictionary containing hidden states and projected embeddings.
        """
        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature": output['hidden_states'], "project": project}

class DetectModel(nn.Module):
    """
    Detection model for identifying classes in images.
    """

    def __init__(self, num_classes=6):
        """
        Initialize the DetectModel.

        Args:
            num_classes (int): Number of classes for detection.
        """
        super(DetectModel, self).__init__()

        self.model = RTDETR()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.agg = AttentionAggregator(256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass for the DetectModel.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor with detection probabilities.
        """
        x = self.model.predict(x)
        x = self.agg(x)
        x = self.fc(x)

        return torch.sigmoid(x)

class SGSeg(nn.Module):
    """
    SGSeg model combining vision, text, and detection models.
    """

    def __init__(self, bert_type, vision_type, project_dim=512):
        """
        Initialize the SGSeg model.

        Args:
            bert_type (str): Type of BERT model.
            vision_type (str): Type of vision model.
            project_dim (int): Dimension of the projection layer.
        """
        super(SGSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type, trust_remote_code=True)
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.detector = DetectModel()

        self.spatial_dim = [7, 14, 28, 56]  # 224*224
        feature_dim = [768, 384, 192, 96]

        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def tokenize(self, captions, device):
        """
        Tokenize the input captions.

        Args:
            captions (list): List of captions.
            device (torch.device): Device to perform computation.

        Returns:
            dict: Dictionary containing input IDs and attention masks.
        """
        input_ids, attention_mask = [], []
        for caption in captions:
            token_output = self.tokenizer.encode_plus(
                caption, padding='max_length',
                max_length=24,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            token, mask = token_output['input_ids'], token_output['attention_mask']
            input_ids.append(token.squeeze(dim=0).tolist())
            attention_mask.append(mask.squeeze(dim=0).tolist())
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        data = {'input_ids': input_ids, 'attention_mask': attention_mask}
        trans = Compose([ToTensord(["input_ids", "attention_mask"])])
        data = trans(data)
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        return {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}

    def seg(self, image, text):
        """
        Perform segmentation.

        Args:
            image (torch.Tensor): Input image tensor.
            text (dict): Tokenized text inputs.

        Returns:
            torch.Tensor: Segmentation output.
        """
        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds, text_project = text_output['feature'], text_output['project']

        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]  # 4 8 16 32 convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]

        reference = text_embeds[-1]

        os32 = image_features[3]
        os16 = self.decoder16(os32, image_features[2], reference)
        os8 = self.decoder8(os16, image_features[1], reference)
        os4 = self.decoder4(os8, image_features[0], reference)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        seg = self.out(os1).sigmoid()

        return seg

    def features(self, image, text):
        """
        Extract features from image and text.

        Args:
            image (torch.Tensor): Input image tensor.
            text (list): List of captions.

        Returns:
            tuple: Tuple containing image features and text embeddings.
        """
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        text = self.tokenize(text, image.device)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds, text_project = text_output['feature'], text_output['project']
        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]  # 4 8 16 32 convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]

        return image_features[-1], text_embeds[-1]

    def detect(self, image):
        """
        Perform detection on the input image.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Detection output.
        """
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        detect = self.detector(image)
        return detect

    def gen(self, image):
        """
        Generate text based on detection results.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            list: List of generated text descriptions.
        """
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        labels = self.detect(image)
        labels = (labels >= 0.5).int().tolist()
        text = []
        for label in labels:
            left, right = label[:3], label[3:]
            left_infected, right_infected = sum(left) > 0, sum(right) > 0
            count_side = int(left_infected) + int(right_infected)
            count_area = sum(label)
            stage1, stage2, stage3 = "", "", ""
            if count_side == 2:
                stage1 = "bilateral pulmonary infection"
            elif count_side == 1:
                stage1 = "unilateral pulmonary infection"
            else:
                text.append("no infection.")
                continue
            if count_area == 1:
                stage2 = "one infected area"
            else:
                ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
                stage2 = ones[count_area] + " infected areas"
            locations = ["upper", "middle", "lower"]
            left_description = " ".join([elem for elem, indicator in zip(locations, left) if indicator == 1]) + " left lung"
            right_description = " ".join([elem for elem, indicator in zip(locations, right) if indicator == 1]) + " right lung"
            if left_infected and right_infected:
                stage3 = left_description + " and " + right_description
            elif left_infected:
                stage3 = left_description
            elif right_infected:
                stage3 = right_description
            else:
                raise ValueError
            gen_text = ", ".join([stage1, stage2, stage3]) + "."
            text.append(gen_text)
        return text

    def forward(self, data, inference=False, image_size={224, 224}):
        """
        Forward pass for the SGSeg model.

        Args:
            data (torch.Tensor or list): Input data.
            inference (bool): Flag to indicate inference mode.
            image_size (tuple): Desired image size.

        Returns:
            tuple: Tuple containing segmentation and detection outputs.
        """
        if inference:
            from monai.transforms import Compose, NormalizeIntensity, Resize, ToTensor, LoadImage
            trans = Compose([
                LoadImage(reader='PILReader', ensure_channel_first=True, image_only=True),
                Resize(spatial_size=image_size, mode='bicubic'),
                NormalizeIntensity(channel_wise=True),
                ToTensor(),
            ])
            image_path = data
            image = trans(image_path).unsqueeze(0)
            text = self.gen(image)

        elif len(data) == 2:
            image, text = data
            label = None
        elif len(data) == 3:
            image, text, label = data
        else:
            raise ValueError(f"Expected data to have 2/3 elements, but got {len(data)}")
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        text = self.tokenize(text, image.device)

        seg = self.seg(image, text)
        detect = self.detector(image)

        return seg, detect
