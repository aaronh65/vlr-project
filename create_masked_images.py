from datetime import datetime
from PIL import Image
import numpy as np
import argparse
import gym
import cv2

from pathlib import Path

def get_mask(rgb, rgb_color, debug=False):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV).flatten()
    #h,s,v = np.float32(hsv_color).flatten()
    low  = np.clip(hsv_color.astype(np.float32)-10, 0, 255).astype(np.uint8)
    #low[1:] = 50
    high = np.clip(hsv_color.astype(np.float32)+10, 0, 255).astype(np.uint8)
    #high[1:] = 255
    mask = cv2.inRange(hsv, low, high)
    return mask

def main(args):
    
    
    skier_rgb = np.uint8([[[214, 92, 92]]])
    flags_rgb = np.uint8([[[66, 72, 200]]])
    rocks_rgb = np.uint8([[[214,214,214]]])
    trees_rgb = np.uint8([[[158,208,101]]]) # light tree

    classes_rgb = {
            'skier': skier_rgb,
            'flags': flags_rgb,
            'rocks': rocks_rgb,
            'trees': trees_rgb,
            }

    classes = classes_rgb.keys()
    for cls in classes:
        cls_path = args.save_root / cls
        cls_path.mkdir(exist_ok=True)

    rgb_path = args.save_root / 'rgb'
    for _path in sorted(rgb_path.glob('*')):
        frame = _path.stem
        rgb = np.array(Image.open(_path))
        for cls, color in classes_rgb.items():
            #debug = cls == 'rocks'
            mask = get_mask(rgb.copy(), color)

            im_path = str(args.save_root / cls / f'{frame}.png')

            cv2.imshow(cls, mask)
            cv2.imwrite(im_path, mask)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', bgr)
        cv2.waitKey(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=Path, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(args)
