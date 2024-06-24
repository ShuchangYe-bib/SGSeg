import os
import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa
import utils.config as config
from engine.wrapper import SGSegWrapper

import pytorch_lightning as pl    
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import warnings
warnings.filterwarnings("ignore")

# Function to parse command line arguments
def get_parser():
    parser = argparse.ArgumentParser(description='Cross-modal Generation')
    parser.add_argument(
        '--config',
        default='./config/base.yaml',
        type=str,
        help='Path to the config file'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse the command line arguments
    args = get_parser()
    
    # Load the configuration from the specified file
    cfg = config.load_cfg_from_cfg_file(args.config)

    torch.manual_seed(42)
    
    # Set the device to GPU if specified, otherwise use CPU
    device = "cuda" if cfg.accelerator == "gpu" else cfg.accelerator
    print("cuda:", torch.cuda.is_available())
    
    # Initialize the training dataset
    ds_train = QaTa(
        ann_path=cfg.train_ann_path,
        root_path=cfg.train_root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode='train'
    )
    
    # Initialize the validation dataset
    ds_valid = QaTa(
        ann_path=cfg.valid_ann_path,
        root_path=cfg.valid_root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode='valid'
    )
    
    # Create data loaders for training and validation datasets
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    # Initialize the model
    model = SGSegWrapper(cfg)
    
    # Load model checkpoint if it exists
    if os.path.exists(cfg.checkpoint_path):
        model.load_from_checkpoint(cfg.checkpoint_path)
    else:
        cfg.checkpoint_path = None
    
    # Setup model checkpointing
    model_ckpt = ModelCheckpoint(
        dirpath=cfg.model_save_path,
        filename=cfg.model_save_filename,
        monitor='monitor_metric',
        save_top_k=1,
        mode='max',
        verbose=True
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        monitor='monitor_metric',
        patience=cfg.patience,
        mode='max'
    )
    
    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        logger=True,
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator, 
        devices=cfg.device,
        callbacks=[model_ckpt, early_stopping],
        enable_progress_bar=True,
        resume_from_checkpoint=cfg.checkpoint_path
    ) 
    
    # Start training
    print('Start training')
    trainer.fit(model, dl_train, dl_valid)
    print('Done training')


