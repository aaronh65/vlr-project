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
from torchvision import transforms

    
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

def eps_greedy(eps, env, q_values):
    if np.random.rand() < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q_values.cpu().squeeze().numpy())   
            
def main(hparams):

    # setup RL environment and burn in data
    env = gym.make('Skiing-v0')
    state = env.reset()
    done = False
    
    train_loader = get_dataloader(hparams, is_train=True)
    burn_in(hparams, env, train_loader)

    if hparams.model == 'base':
        model = DQNBase(3)
    # else:
    #     model = AutoEncoder.load_from_checkpoint(args.ae_path)

    if args.cuda:
        model.cuda()
    
    criterion = torch.nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(list(model.parameters())) 
    gamma = 0.99
    print_freq = 10
    val_freq = 250
    # ignore for now
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 160)),
            transforms.ToTensor()
            ])
    for i in range(hparams.max_epochs):
        
        state = env.reset()
        done = False
        for batch_nb, batch in tqdm(enumerate(train_loader)):
            model.eval()
            for t in range(hparams.rollout_steps_per_iteration):
                if done:
                    done = False
                    state = env.reset()
                
                obs = transform(env.render(mode='rgb_array')).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(obs)
                action = eps_greedy(eps=0.1, env=env, q_values=q_values)
                new_state, reward, done, info = env.step(action) # change this to eps greedy 
                train_loader.dataset.add_experience(state, action, reward, done)
                state = new_state
            
            model.train()
            
            # overloaded vars
            optim.zero_grad()
            if args.cuda:
                for i, item in enumerate(batch[:-1]): # excluding info
                    batch[i] = item.cuda()
            _state, _action, _reward, _next_state, _done, _info = batch

            # retrieve Q(s,a)
            Qs = model(_state) # N,3
            Qsa = Qs[torch.arange(_state.shape[0]).long(), _action.long()]

            # retrieve nQ(s,a)
            with torch.no_grad():
                nQs = model(_next_state) # N,3
                nQsa, _ = torch.max(nQs, dim=-1)
            
            target = _reward * (1 - _done) * gamma * nQsa

            batch_loss = criterion(Qsa, target)
            loss = batch_loss.mean()
            loss.backward()
            optim.step()

            if batch_nb % print_freq == 0:
                tqdm.write("Loss: {}".format(loss.item()))

            if batch_nb % val_freq == 0:
                model.eval()
                total_reward = 0
                for i in range(hparams.num_eval_episodes):
                    val_done = False
                    val_state = env.reset()
                    while not val_done:
                        if args.render:
                            env.render()
                        
                        obs = transform(env.render(mode='rgb_array')).unsqueeze(0)
                        with torch.no_grad():
                            q_values = model(obs)
                        val_action = eps_greedy(eps=0, env=env, q_values=q_values)
                        # change this to eps greedy 
                        val_new_state, val_reward, val_done, val_info = env.step(val_action) 
                        val_state = val_new_state
                        total_reward += val_reward
                total_reward /= hparams.num_eval_episodes
                tqdm.write("Mean val reward: {}".format(total_reward))

                done = True # reset at beginning of next train iter
                model.train()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # program args
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--render', action='store_true')

    # DRL args
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--burn_steps', type=int, default=2000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--rollout_steps_per_iteration', type=int, default=20)
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--ae_path', type=str, default='checkpoints/autoencoder/20210423_184757/epoch=9.ckpt')
    args = parser.parse_args()

    main(args)
