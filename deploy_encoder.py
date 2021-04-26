import argparse
import torch
import gym
from pathlib import Path
from dataloader import get_dataloader
from autoencoder import AutoEncoder

DISPLAY=True


def evaluate_on_dataset(model, args):
    model.eval()
    model.cuda()

    dataloader = get_dataloader(args, is_train=True)

    for batch_nb, batch in enumerate(dataloader):
        rgb = batch['rgb'].cuda()

        skier = batch['skier'].cuda()
        flags = batch['flags'].cuda()
        rocks = batch['rocks'].cuda()
        trees = batch['trees'].cuda()

        gt_masks = torch.cat((skier, flags, rocks, trees), dim=1)

        pred_masks, latent = model(rgb)

        all_images = model.make_visuals(rgb, gt_masks, pred_masks)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/20210413_182405')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--model_dir', type=str, 
            default='/home/aaron/workspace/vlr/vlr-project/checkpoints/20210423_184757')
    args = parser.parse_args()

    ckpts = sorted(list(Path(args.model_dir).glob('*.ckpt')))
    model = AutoEncoder.load_from_checkpoint(str(ckpts[-1]))
    
    #evaluate_on_dataset(model)
