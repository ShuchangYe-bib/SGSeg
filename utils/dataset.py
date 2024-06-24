import json
import os
import torch
import pandas as pd
from monai.transforms import (
    Compose, NormalizeIntensityd, RandZoomd,
    Resized, ToTensord, LoadImaged, EnsureChannelFirstd
)
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class QaTa(Dataset):
    """
    QaTa dataset class for handling image-caption datasets.
    """

    def __init__(self, data=None, ann_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224,224], inference=False):
        """
        Initialize the QaTa dataset.

        Args:
            data (pd.DataFrame): DataFrame containing dataset information.
            ann_path (str): Path to the annotation file.
            root_path (str): Root path for images and masks.
            tokenizer (str): Pretrained tokenizer name or path.
            mode (str): Mode of the dataset, one of 'train', 'valid', or 'test'.
            image_size (list): Desired image size.
            inference (bool): Whether the dataset is for inference.
        """
        super(QaTa, self).__init__()

        self.mode = mode
        self.inference = inference

        if data is not None:
            self.data = data
            self.image_list = list(self.data['Image'])
            self.caption_list = list(self.data['Description'])
            self.label_list = [] if self.inference else list(self.data["Pseudo"])

        elif ann_path.split(".")[-1] == "csv":
            with open(ann_path, 'r') as f:
                self.data = pd.read_csv(f)
            self.image_list = list(self.data['Image'])
            self.caption_list = list(self.data['Description'])
            self.label_list = [] if self.inference else list(self.data["Pseudo"])

            if mode == 'train':
                split_point = int(0.8 * len(self.image_list))
                self.image_list = self.image_list[:split_point]
                self.caption_list = self.caption_list[:split_point]
                self.label_list = [] if self.inference else self.label_list[:split_point]

            elif mode == 'valid':
                split_point = int(0.8 * len(self.image_list))
                self.image_list = self.image_list[split_point:]
                self.caption_list = self.caption_list[split_point:]
                self.label_list = [] if self.inference else self.label_list[split_point:]

        elif ann_path.split(".")[-1] == "json":
            self.image_list = []
            self.caption_list = []
            self.label_list = []
            annotations = json.loads(open(ann_path, 'r').read())[mode]
            for anno in annotations:
                self.image_list.append(anno["image_path"])
                self.caption_list.append(anno["caption"])
                if not self.inference:
                    self.label_list.append(anno["pseudo_label"])

        self.root_path = root_path
        self.image_size = image_size

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing image, ground truth, and labels.
        """
        trans = self.transform(self.image_size)

        image_path = os.path.join(self.root_path, 'images', self.image_list[idx].replace('mask_', ''))
        gt_path = os.path.join(self.root_path, 'masks', self.image_list[idx])
        text = self.caption_list[idx]

        if not self.inference:
            label = self.label_list[idx]

        data = {'image': image_path, 'gt': gt_path}
        data = trans(data)

        image, gt = data['image'], data['gt']
        gt = torch.where(gt == 255, 1, 0)

        if not self.inference:
            label = torch.tensor(label, dtype=torch.float32).to(image.device)
            return ([image, text, label], gt)
        else:
            return ([image, text], gt)
        
    def transform(self, image_size=[224,224]):
        """
        Define the transformation pipeline.

        Args:
            image_size (list): Desired image size.

        Returns:
            Compose: Composition of transformations.
        """
        if self.mode == 'train':
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader', image_only=False),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])
        else:
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader', image_only=False),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])

        return trans


