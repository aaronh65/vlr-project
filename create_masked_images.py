from datetime import datetime
from PIL import Image
import numpy as np
import argparse
import gym
import cv2

from pathlib import Path

def get_mask(rgb, hsv_color, debug=False):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    #hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV).flatten()
    h,s,v = np.float32(hsv_color).flatten()
    #low  = np.clip((h-10, s-1, v-1), 0, 255).astype(np.uint8)
    #high = np.clip((h+10, s+1, v+1), 0, 255).astype(np.uint8)
    low  = np.clip(hsv_color.astype(np.float32)-1, 0, 255).astype(np.uint8)
    high = np.clip(hsv_color.astype(np.float32)+1, 0, 255).astype(np.uint8)
    mask = cv2.inRange(hsv, low, high)
    return mask

def main(args):
    
    
    # retrieve HSV colors
    skier_rgb = np.uint8([[[214, 92, 92]]])
    flags_rgb = np.uint8([[[66, 72, 200]]])
    rocks_rgb = np.uint8([[[214,214,214], [192,192,192]]])
    trees_rgb = np.uint8([[[158,208,101], [72, 160, 72], [110, 156, 66], [82,126,45]]]) # light tree
    classes_rgb = {
            'skier': skier_rgb,
            'flags': flags_rgb,
            'rocks': rocks_rgb,
            'trees': trees_rgb,
            }

    classes_hsv = {}
    for cls, rgbs in classes_rgb.items():
        classes_hsv[cls] = cv2.cvtColor(rgbs, cv2.COLOR_RGB2HSV)[0]

    for cls in classes_hsv.keys():
        cls_path = args.save_root / cls
        cls_path.mkdir(exist_ok=True)

    # iterate through images
    rgb_path = args.save_root / 'rgb'
    for _path in sorted(rgb_path.glob('*')):
        frame = _path.stem
        rgb = np.array(Image.open(_path))
        h,w,c = rgb.shape
        for cls, colors in classes_hsv.items():
            mask = np.zeros((h,w)).astype(bool)
            for color in colors:
                temp_mask = get_mask(rgb.copy(), color)
                temp_mask = temp_mask.astype(bool)
                mask = np.logical_or(mask, temp_mask)
            mask = np.uint8(mask) * 255
            cv2.imshow(cls, mask)

            im_path = str(args.save_root / cls / f'{frame}.png')
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
