import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from create_masked_images import get_hsv_colors, get_mask


class SkierDataset(Dataset):
    def __init__(self, episodes):
        self.frames = list()
        for ep_path in episodes:
            rgb_path = ep_path / 'rgb'
            self.frames.extend(sorted(list(rgb_path.glob('*'))))
        self.classes_hsv = get_hsv_colors() # object to hsv color dict

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        frame = self.frames[i]
        rgb = np.array(Image.open(frame))
        h,w,c = rgb.shape
        cv2.imshow('rgb', rgb)
        
        masks = list()
        for cls, colors in self.classes_hsv.items():
            mask = np.zeros((h,w)).astype(bool)
            for color in colors:
                temp_mask = get_mask(rgb.copy(), color)
                temp_mask = temp_mask.astype(bool)
                mask = np.logical_or(mask, temp_mask)
            mask = np.uint8(mask) * 255
            masks.append(mask)
        masks = np.hstack(masks)
        cv2.imshow('masks', masks)
        cv2.waitKey(10)

        return frame

def get_dataloader(args, is_train=False):
    data_dir = Path(args.dataset_dir)

    paths = sorted(list(data_dir.glob('*')))

    episodes = list()
    for i, path in enumerate(paths):
        if is_train and i % 10 != 0:
            episodes.append(path)
        elif not is_train and i % 10 == 0:
            episodes.append(path)

    dataset = SkierDataset(episodes)
    return dataset
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/20210413_182405')
    args = parser.parse_args()
    
    dataset = get_dataloader(args, is_train=True)
    
    print(f'train dataset has {len(dataset)} samples')
    for data in dataset:
        pass
        #print(data)
