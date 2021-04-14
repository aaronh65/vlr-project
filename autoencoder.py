import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        pass

    def forward(self, input):
        pass

    def training_step(self, batch, batch_nb):
        pass
    def validation_step(self, batch, batch_nb):
        pass

    def train_dataloader(self):
        pass
    def val_dataloader(self):
        pass

    def configure_optimizers(self):
        pass

def main(args):
    pass

if __name__ == '__main__':
    pass
