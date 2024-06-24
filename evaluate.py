import argparse
from engine.wrapper import SGSegWrapper

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.dataset import QaTa
import utils.config as config

import warnings 
warnings.filterwarnings('ignore') 

# Custom argparse action to append version info to the model
class Version(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, '-v{}'.format(values))

# Function to parse command line arguments
def get_parser():
    parser = argparse.ArgumentParser(description='Cross-modal Generation')
    parser.add_argument(
        '--config',
        default='./config/base.yaml',
        type=str,
        help='Path to the config file'
    )
    parser.add_argument(
        '--v',
        default='',
        type=str,
        action=Version,
        help='Model version'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse the command line arguments
    args = get_parser()
    
    # Load the configuration from the specified file
    cfg = config.load_cfg_from_cfg_file(args.config)
    
    # Set the device to GPU if specified, otherwise use CPU
    device = "cuda" if cfg.accelerator == "gpu" else cfg.accelerator

    # Initialize the model
    model = SGSegWrapper(cfg)
    
    # Load the model checkpoint
    checkpoint = torch.load(
        './save_model/sgseg{}.ckpt'.format(args.v),
        map_location=device
    )["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    
    # Initialize the test dataset
    ds_test = QaTa(
        ann_path=cfg.test_ann_path,
        root_path=cfg.test_root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode='test'
    )
    
    # Create a data loader for the test dataset
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=8
    )
    
    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(accelerator='gpu', devices=1) 
    
    # Set the model to evaluation mode
    model.eval()
    
    # Run the test
    trainer.test(model, dl_test)


