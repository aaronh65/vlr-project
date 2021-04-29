
from itertools import islice
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
    def __init__(self, args, is_train=True):
        #self.frames = list()
        #for ep_path in episodes:
        #    rgb_path = ep_path / 'rgb'
        #    self.frames.extend(sorted(list(rgb_path.glob('*'))))
        self.args = args
        self.epoch_len = args.epoch_len*args.batch_size if is_train else args.num_eval_episodes
        self.buffer_len = args.buffer_len
        self.buffer = deque()
        self.is_train = is_train


        self.classes_hsv = get_hsv_colors() # object to hsv color dict
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])

        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.dones = deque()

        self.step = 0
        self.indices = np.random.choice(np.arange(self.buffer_len), size=self.epoch_len)

    def __len__(self):
        return self.epoch_len

    def add_experience(self, state, action, reward, done):

        # pop if at buffer limit
        if len(self.states) > self.buffer_len:
            self.states.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.dones.popleft()

        h,w,c = state.shape
        state[:h//10] = 0
        #state = np.array(Image.fromarray(state).convert("L"))
        #state = np.expand_dims(state, -1)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def _get_history(self, i):
        states = [self.states[i]]
        condition = False # add zero image if true
        for dt in range(1, self.args.history_size):
            _i = i - dt
            condition = condition or _i <= 0
            condition = condition or self.dones[_i]
            add = np.zeros(states[0].shape) if condition else self.states[_i]
            states.append(np.uint8(add))
        states = np.stack(states)
        return states

    def _get_state(self, i):
        state1 = self._get_history(i)
        state2 = self._get_history(i-1)
        #combined = np.maximum(state1, state2) # N,3,H,W
        combined = state1
        result = list()
        for i in range(combined.shape[0]):
            #print(combined[i])
            img = Image.fromarray(combined[i]).convert("L")
            img = np.expand_dims(np.array(img), -1)
            result.append(self.transform(img))
        result = torch.cat(result, dim=0)
        #print(result.shape)

        return result
        
    def __getitem__(self, i):

        #print(self.step)
        if not self.is_train:
            return torch.ones(5)

        # retrieve index
        #if self.step >= self.epoch_len - 1:
        #    self.indices = np.random.choice(np.arange(self.buffer_len), size=self.epoch_len)
        #i = self.indices[i] % len(self.states)
        i = np.random.choice(np.arange(len(self.states)))
        i = max(i, 1)

        state = self._get_state(i)
        action = torch.FloatTensor([self.actions[i]])
        reward = torch.FloatTensor([self.rewards[i]])
        done = torch.FloatTensor([self.dones[i]])
        next_index = i+1 if not self.dones[i] else i
        next_index = min(next_index, len(self.states)-1)
        next_state = self._get_state(next_index)

        info = {}

        self.step += 1
        return state, action, reward, next_state, done, info

def get_dataloader(args, is_train=False):
    dataset = SkierRLDataset(args, is_train)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return dataloader
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    dataloader = get_dataloader(args, is_train=True)
    
