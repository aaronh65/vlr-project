from datetime import datetime
from PIL import Image
import numpy as np
import argparse
import gym
import cv2

from pathlib import Path

def main(args):
    
    classes = ['trees']
    for cls in classes:
        cls_path = args.save_root / cls
        cls_path.mkdir(exist_ok=True)

    rgb_path = args.save_root / 'rgb'
    light = (88,51,82)
    dark = (93,64,49)
    for _path in sorted(rgb_path.glob('*')):
        img = np.array(Image.open(_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        print(np.amax(img))
        print(np.amin(img))
        mask = cv2.inRange(img, light, dark)
        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.waitKey(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=Path, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(args)
