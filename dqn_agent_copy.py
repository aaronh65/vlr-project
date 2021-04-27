from datetime import datetime
from pathlib import Path
from models import DQNModel, DQNBase
from dqn_dataloader import get_dataloader
from utils import spatial_norm
from autoencoder import AutoEncoder

import gym
# import wandb
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm

DISPLAY=True
    
def burn_in(hparams, env, train_loader):
    state = env.reset()
    done = False
    for t in range(hparams.burn_steps):
        if done:
            done = False
            state = env.reset()

        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action) # change this to eps greedy 
        train_loader.dataset.add_experience(state,action,reward,done)

        state = new_state
            
def main(hparams):
    #if hparams.log:
    #    logger = WandbLogger(save_dir=hparams.save_dir, project='vlr-project')
    #else:
    #    logger = False
    logger = False

    #checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3)
    # setup RL environment and burn in data
    env = gym.make('Skiing-v0')
    state = env.reset()
    done = False
    
    train_loader = get_dataloader(hparams, is_train=True)
    burn_in(hparams, env, train_loader)

    if hparams.model == 'base':
        model = DQNBase(3)
    else:
        print("not done yet")
    
    criterion = torch.nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(list(model.parameters())) 
    gamma = 0.99
    print_freq = 10
    val_freq = 250
    # ignore for now
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)

    for i in range(hparams.max_epochs):
        state = env.reset()
        done = False
        print(len(train_loader))
        for batch_nb, batch in tqdm(enumerate(train_loader)):
            for t in range(hparams.rollout_steps_per_iteration):
                if done:
                    done = False
                    state = env.reset()

                action = env.action_space.sample()
                new_state, reward, done, info = env.step(action) # change this to eps greedy 
                train_loader.dataset.add_experience(state, action, reward, done)
                state = new_state

            # overloaded vars
            optim.zero_grad()
            _state, action, reward, next_state, _done, info = batch

            # retrieve Q(s,a)
            Qs = model(_state) # N,3
            Qsa = Qs[torch.arange(_state.shape[0]).long(), action.long()]

            # retrieve nQ(s,a)
            with torch.no_grad():
                nQs = model(next_state) # N,3
                nQsa, _ = torch.max(nQs, dim=-1)
            
            target = reward * (1-_done) * gamma * nQsa

            batch_loss = criterion(Qsa, target)
            loss = batch_loss.mean()
            loss.backward()
            optim.step()

            if batch_nb % print_freq == 0:
                print("Loss: {}".format(loss.item()))

            if batch_nb % val_freq == 0:
                # val here
                val_iters = 10
                total_reward = 0
                for i in range(val_iters):
                    val_done = False
                    val_state = env.reset()
                    while not val_done:
                        val_action = env.action_space.sample()
                        val_new_state, val_reward, val_done, info = env.step(val_action) # change this to eps greedy 
                        state = new_state
                        total_reward += val_reward
                
                total_reward /= 10
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--burn_steps', type=int, default=2000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--rollout_steps_per_iteration', type=int, default=20)
    parser.add_argument('--model', type=str, default='base')
    args = parser.parse_args()

    main(args)
