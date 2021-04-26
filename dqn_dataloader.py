
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from create_masked_images import get_hsv_colors, get_mask
from collections import deque

SHOW = False

class SkierRLDataset(Dataset):
    def __init__(self, args, is_train=False):
        #self.frames = list()
        #for ep_path in episodes:
        #    rgb_path = ep_path / 'rgb'
        #    self.frames.extend(sorted(list(rgb_path.glob('*'))))
        self.epoch_len = args.epoch_len
        self.buffer_len = args.buffer_len
        self.buffer = deque()

        self.classes_hsv = get_hsv_colors() # object to hsv color dict
        self.transform = transforms.Compose([
            transforms.Resize((256, 160)),
            transforms.ToTensor()
            ])

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        print(self.buffer)
        #res = {}
        #frame = self.frames[i]

        #rgb = np.array(Image.open(frame))
        #
        #h,w,c = rgb.shape
        #masks = list()
        #for cls, colors in self.classes_hsv.items():
        #    mask = np.zeros((h,w)).astype(bool)
        #    for color in colors:
        #        temp_mask = get_mask(rgb.copy(), color)
        #        temp_mask = temp_mask.astype(bool)
        #        mask = np.uint8(np.logical_or(mask, temp_mask))*255
        #    mask_image = Image.fromarray(mask)
        #    mask_tensor = self.transform(mask_image)
        #    res[cls] = mask_tensor
        #    masks.append(np.uint8(mask) * 255)
        #masks = np.hstack(masks)

        #rgb_tensor = self.transform(Image.fromarray(rgb))
        #res['rgb']  = rgb_tensor

        return torch.ones(5)

def get_dataloader(args, is_train=False):
    dataset = SkierRLDataset(args, is_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return dataloader
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    dataloader = get_dataloader(args, is_train=True)
    
