from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models import Encoder, Decoder
from dataloader import get_dataloader

import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse


class AutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=4)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input):
        return torch.ones(self.hparams.batch_size)

    def training_step(self, batch, batch_nb):
        rgb = batch['rgb']
        skier = batch['skier']
        flags = batch['flags']
        rocks = batch['rocks']
        trees = batch['trees']

        gt_masks = torch.cat((skier, flags, rocks, trees), dim=1)
        latent = self.encoder(rgb)
        pred_masks = self.decoder(latent)

        loss = self.criterion(pred_masks, gt_masks)
        if self.logger != None:
            self.logger.log_metrics({'train/loss': loss.mean().item()}, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        rgb = batch['rgb']
        skier = batch['skier']
        flags = batch['flags']
        rocks = batch['rocks']
        trees = batch['trees']

        gt_masks = torch.cat((skier, flags, rocks, trees), dim=1)
        latent = self.encoder(rgb)
        pred_masks = self.decoder(latent)

        val_loss = self.criterion(pred_masks, gt_masks)
        if self.logger != None:
            self.logger.log_metrics({'val/loss': val_loss.mean().item()}, self.global_step)

        return {'val_loss': val_loss}

    def train_dataloader(self):
        return get_dataloader(self.hparams, is_train=True)

    def val_dataloader(self):
        return get_dataloader(self.hparams, is_train=False)

    def configure_optimizers(self):
        # add in lr from hparams if default adam sucks
        optim = torch.optim.Adam(list(self.encoder.parameters())) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)
        return [optim], [scheduler]

def main(hparams):
    if hparams.log:
        logger = WandbLogger(save_dir=hparams.save_dir, project='vlr-project')
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3, monitor='val_loss')
    model = AutoEncoder(hparams)
    trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            checkpoint_callback=checkpoint_callback,
            enable_pl_optimizer=False,
            logger=logger,
            distributed_backend='dp'
            )
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/20210413_182405')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(exist_ok=True, parents=True)
    args.save_dir = str(save_dir)
    main(args)
