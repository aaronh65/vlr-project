from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from models import Encoder
from dataloader import get_dataloader
import torch
import pytorch_lightning as pl
import argparse


class AutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = Encoder()

    def forward(self, input):
        return torch.ones(self.hparams.batch_size)

    def training_step(self, batch, batch_nb):
        rgb = batch['rgb']
        skier = batch['skier']

        print(rgb.shape)

        loss = torch.ones(self.hparams.batch_size).mean()
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        val_loss = torch.ones(self.hparams.batch_size).mean()
        return {'val_loss': val_loss}

    def train_dataloader(self):
        return get_dataloader(self.hparams, is_train=True)

    def val_dataloader(self):
        return get_dataloader(self.hparams, is_train=False)

    def configure_optimizers(self):
        optim = torch.optim.Adam(list(self.encoder.parameters())) # add in lr from hparams if default adam sucks
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)
        return [optim], [scheduler]

def main(hparams):
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3, monitor='val_loss')
    model = AutoEncoder(hparams)
    trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            checkpoint_callback=checkpoint_callback,
            enable_pl_optimizer=False,
            distributed_backend='dp'
            )
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/20210413_182405')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(exist_ok=True, parents=True)
    args.save_dir = str(save_dir)
    main(args)
