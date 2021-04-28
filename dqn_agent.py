from datetime import datetime
from pathlib import Path
from models import DQNModel, DQNBase
from dqn_dataloader import get_dataloader
from utils import spatial_norm
from autoencoder import AutoEncoder
import copy

import gym
# import wandb
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageDraw

    
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

def visualize(state, action, reward, next_state, done, meta):
    #N = min(state.shape[0], 8)
    N = 8
    images = list()
    Q, nQ, meaning, next_action = meta['Q'], meta['nQ'], meta['meaning'], meta['next_action']
    for n in range(N):
        _state = state[n].cpu().numpy().transpose(1,2,0)
        _state = np.uint8(_state*255)
        _next_state = next_state[n].cpu().numpy().transpose(1,2,0)
        _next_state = np.uint8(_next_state*255)
        text = np.uint8(np.zeros(_next_state.shape))
        text = Image.fromarray(text)
        draw = ImageDraw.Draw(text)

        _action = int(action[n].item())
        draw.text((5,10), f'action \t= {meaning[_action]}', (255,255,255))
        draw.text((5,20), f'Q(s,a) \t= {Q[n].item():.5f}', (255,255,255))
        draw.text((5,30), f'reward \t= {reward[n].item()}', (255,255,255))
        draw.text((5,40), f'nQ(s,a) \t= {nQ[n].item():.5f}', (255,255,255))
        _next_action = int(next_action[n].item())
        draw.text((5,50), f'naction \t= {meaning[_next_action]}', (255,255,255))
        draw.text((5,60), f'done \t= {bool(done[n].item())}', (255,255,255))
        combined = np.hstack((_state, _next_state, text))
        images.append(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    left = np.vstack(images[:4])
    right = np.vstack(images[4:])
    images = np.hstack((left,right))
    cv2.imshow('debug', images)
    cv2.waitKey(10)



def eps_greedy(eps, env, q_values):
    if np.random.rand() < eps:
        return env.action_space.sample()
    else:
        vals = q_values.cpu().squeeze().numpy()
        return np.argmax(vals)
            
def main(hparams):

    # setup RL environment and burn in data

    env = gym.make('Pong-v0')
    #env = gym.make('Skiing-v0')
    #env = gym.make('CartPole-v0')
    meaning = env.get_action_meanings()
    state = env.reset()
    done = False
    
    train_loader = get_dataloader(hparams, is_train=True)
    burn_in(hparams, env, train_loader)

    if hparams.model == 'base':
        model = DQNBase(env.action_space.n)
    # else:
    #     model = AutoEncoder.load_from_checkpoint(args.ae_path)
    model_t = copy.deepcopy(model)
    model_t.eval()

    if args.cuda:
        model.cuda()
        model_t.cuda()


    criterion = torch.nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(list(model.parameters())) 
    gamma = 0.99
    print_freq = 50
    val_freq = 500
    step = 0
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
                
                if args.render:
                    env.render()
                #obs = transform(env.render(mode='rgb_array')).unsqueeze(0)
                model_input = transform(state).unsqueeze(0)
                if args.cuda:
                    model_input = model_input.cuda()
                with torch.no_grad():
                    q_values = model(model_input)
                action = eps_greedy(eps=0.3, env=env, q_values=q_values)
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
            Qsa = torch.gather(Qs, -1, _action.long())
            #Qsa = Qs[torch.arange(_state.shape[0]).long(), _action.long()]

            # retrieve nQ(s,a)
            with torch.no_grad():
                nQs = model_t(_next_state) # N,3
                nQsa, _next_action = torch.max(nQs, dim=-1, keepdim=True)
                
            target = _reward * (1 - _done) * gamma * nQsa

            batch_loss = criterion(Qsa, target)
            loss = batch_loss.mean()
            loss.backward()
            optim.step()

            meta = {'Q': Qsa, 'nQ': nQsa, 'meaning': meaning, 'next_action': _next_action}


            if batch_nb % print_freq == 0:
                tqdm.write("Loss: {}".format(loss.item()))
                visualize(_state, _action, _reward, _next_state, _done, meta)

            if batch_nb % val_freq == 0 and batch_nb != 0:
                model.eval()
                total_reward = 0
                for i in range(hparams.num_eval_episodes):
                    val_done = False
                    val_state = env.reset()
                    while not val_done:
                        if args.render:
                            env.render()
                        
                        obs = transform(env.render(mode='rgb_array')).unsqueeze(0)
                        if args.cuda:
                            obs = obs.cuda()
                        with torch.no_grad():
                            q_values = model(obs)
                        val_action = eps_greedy(eps=0, env=env, q_values=q_values)
                        #tqdm.write(str(val_action))
                        # change this to eps greedy 
                        val_new_state, val_reward, val_done, val_info = env.step(val_action) 
                        val_state = val_new_state
                        total_reward += val_reward
                total_reward /= hparams.num_eval_episodes
                tqdm.write("Mean val reward: {}".format(total_reward))

                done = True # reset at beginning of next train iter
                model.train()
                step += 1

                if step % args.target_update_freq == 0:
                    model_t.load_state_dict(model.state_dict())
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # program args
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--render', action='store_true')

    # DRL args
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--burn_steps', type=int, default=1000)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=5)
    parser.add_argument('--rollout_steps_per_iteration', type=int, default=20)
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--ae_path', type=str, default='checkpoints/autoencoder/20210423_184757/epoch=9.ckpt')
    args = parser.parse_args()

    main(args)
