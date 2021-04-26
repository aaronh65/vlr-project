from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models import DQNModel, DQNBase
from dqn_dataloader import get_dataloader
from utils import spatial_norm
from autoencoder import AutoEncoder

import gym
import wandb
import cv2
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import argparse

DISPLAY=True

class DQNAgent(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #self.autoencoder = AutoEncoder.load_from_checkpoint(hparams.autoencoder_path)
        #self.autoencoder.eval()


        # setup RL environment and burn in data
        self.env = gym.make('Skiing-v0')
        self.model = DQNBase(4)

    def forward(self, rgb):
        #latent = self.res_encoder(rgb)
        #pred_masks = self.res_decoder(latent)

        #return pred_masks, latent
        return None


    def training_step(self, batch, batch_nb):

        #batch is (s,a,r,ns,etc.)

        #train DQN model on TD loss

        #rollout environment to add to train dataloader

        loss = torch.ones(5)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        self.val_loader.dataset.buffer.append(batch_nb)

        val_loss = torch.ones(5)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()
                results[key].append(metrics[key].mean().item())

        summary = {key: np.mean(val) for key, val in results.items()}
        if self.logger != None:
            self.logger.log_metrics(summary, self.global_step)
        return summary

    def train_dataloader(self):
        self.train_loader = get_dataloader(self.hparams, is_train=True)
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = get_dataloader(self.hparams, is_train=False)
        return self.val_loader

    def configure_optimizers(self):
        # add in lr from hparams if default adam sucks
        optim = torch.optim.Adam(list(self.model.parameters())) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)
        return [optim], [scheduler]
    
def main(hparams):
    #if hparams.log:
    #    logger = WandbLogger(save_dir=hparams.save_dir, project='vlr-project')
    #else:
    #    logger = False
    logger = False

    #checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3)
    model = DQNAgent(hparams)
    trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            #gpus=hparams.gpus,
            #checkpoint_callback=checkpoint_callback,
            enable_pl_optimizer=False,
            logger=logger,
            #distributed_backend='dp'
            )
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)

    args = parser.parse_args()

    main(args)
