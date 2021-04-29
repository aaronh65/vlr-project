import time
from datetime import datetime
from collections import deque
from pathlib import Path
from models import DQNModel, DQNBase
from dqn_dataloader import get_dataloader
from utils import spatial_norm
from autoencoder import AutoEncoder
import copy

import gym
import wandb
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

def visualize(state, action, reward, next_state, done, meta, args):
    #N = min(state.shape[0], 8)
    N = 8
    images = list()
    Q, nQ, meaning, next_action = meta['Q'], meta['nQ'], meta['meaning'], meta['next_action']
    batch_loss = meta['batch_loss'].flatten()
    for n in range(N):
        _state = state[n,:1].cpu().numpy().transpose(1,2,0)
        _state = np.uint8(_state*255)
        _state = np.tile(_state, (1,1,3))
        _next_state = next_state[n,:1].cpu().numpy().transpose(1,2,0)
        _next_state = np.uint8(_next_state*255)
        _next_state = np.tile(_next_state, (1,1,3))
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
        draw.text((5,60), f'loss \t= {batch_loss[n].item():.5f}', (255,255,255))
        draw.text((5,70), f'done \t= {bool(done[n].item())}', (255,255,255))
        combined = np.hstack((_state, _next_state, text))
        images.append(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    left = np.vstack(images[:4])
    right = np.vstack(images[4:])
    images = np.hstack((left,right))
    if args.render:
        cv2.imshow('debug', images)
        cv2.waitKey(100)
    return images



def eps_greedy(eps, env, q_values):
    if np.random.rand() < eps:
        return env.action_space.sample()
    else:
        vals = q_values.cpu().squeeze().numpy()
        return np.argmax(vals)
            
def main(hparams):

    if hparams.log:
        wandb.init('vlr-project-dqn')
        wandb.config.update(hparams)
    # setup RL environment and burn in data

    #env = gym.make('Pong-v0')
    #env = gym.make('VideoPinball-v0')
    env = gym.make('Breakout-v0')
    #env = gym.make('Skiing-v0')
    #env = gym.make('CartPole-v0')
    meaning = env.get_action_meanings()
        
    train_loader = get_dataloader(hparams, is_train=True)
    burn_in(hparams, env, train_loader)

    if hparams.model == 'base':
        model = DQNBase(env.action_space.n, hparams.history_size)
    # else:
    #     model = AutoEncoder.load_from_checkpoint(args.ae_path)
    model_t = copy.deepcopy(model)
    model_t.eval()

    if hparams.cuda:
        model.cuda()
        model_t.cuda()

    criterion = torch.nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(list(model.parameters())) 
    gamma = 0.99
    print_freq = 50
    # ignore for now
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])

    history = deque()
    state = env.reset()
    done = True
    num_evals = 0
    step = 0
    best_val = -float('inf')
    for epoch in range(hparams.max_epochs):
        
        eps = 1 - epoch / hparams.max_epochs
        eps = max(eps, 0.1)
        for batch_nb, batch in tqdm(enumerate(train_loader)):

            ###########
            # ROLLOUT #
            ###########

            model.eval()
            for t in range(hparams.rollout_steps_per_iteration):
                if done: # reset if needed
                    done = False
                    history = deque([torch.zeros(1,224,224)]*hparams.history_size)
                    state = env.reset()
                    last_action = env.action_space.sample()

                if hparams.render:
                    env.render()

                # prepare inputs
                if step % hparams.k == 0:
                    _input = np.array(Image.fromarray(state).convert("L"))
                    history.pop()
                    history.appendleft(transform(_input))
                    input = list(history)
                    input = torch.cat(input,dim=0).unsqueeze(0)
                    if hparams.cuda:
                        input = input.cuda()

                    # retrieve action and step
                    with torch.no_grad():
                        q_values = model(input)
                    action = eps_greedy(eps=eps, env=env, q_values=q_values)
                else:
                    action = last_action

                new_state, reward, done, info = env.step(action) 
                reward = min(max(reward,-1), 1) # clip to [-1,1]

                # add data
                train_loader.dataset.add_experience(state, action, reward, done)
                state = new_state
                last_action = action
            
            ############
            # TRAINING #
            ############

            metrics = {}
            model.train()
            if hparams.cuda:
                for i, item in enumerate(batch[:-1]): # excluding info
                    batch[i] = item.cuda()
            _state, _action, _reward, _next_state, _done, _info = batch
            
            # retrieve Q(s,a) and Q(ns, na)
            Qs = model(_state) # N,3
            Qsa = torch.gather(Qs, -1, _action.long())
            with torch.no_grad():
                nQs = model_t(_next_state) # N,3
                nQsa, _next_action = torch.max(nQs, dim=-1, keepdim=True)
                
            target = _reward + (1 - _done) * gamma * nQsa
            batch_loss = criterion(Qsa, target)
            loss = batch_loss.mean()
            metrics['train_loss'] = loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % print_freq == 0 and not hparams.log:
                tqdm.write("Loss: {}".format(loss.item()))

            if step % hparams.target_update_freq == 0:
                    model_t.load_state_dict(model.state_dict())

            if step % hparams.visual_freq == 0 and (hparams.log or hparams.render):
                meta = {
                    'Q': Qsa, 
                    'nQ': nQsa, 
                    'meaning': meaning, 
                    'next_action': _next_action, 
                    'batch_loss': batch_loss
                }
                images = visualize(_state, _action, _reward, _next_state, _done, meta, hparams)
                metrics['image'] = wandb.Image(images)

            if hparams.log:
                wandb.log(metrics, step)


            ##############
            # EVALUATION #
            ##############

            if step % hparams.val_freq == 0:
                
                tqdm.write(f'epoch {epoch} step {batch_nb} validation')
                model.eval()
                total_reward = 0
                val_done = False
                for i in range(hparams.num_eval_episodes):

                    val_done = False
                    val_history = deque([torch.zeros(1,224,224)]*hparams.history_size)
                    val_state = env.reset()
                    val_step = 0
                    while not val_done:

                        if hparams.render:
                            env.render()
                        
                        _val_state = Image.fromarray(val_state).convert("L")
                        _val_state = np.expand_dims(np.array(_val_state),-1)
                        val_history.pop()
                        val_history.appendleft(transform(_val_state))
                        val_input = list(val_history)
                        val_input = torch.cat(val_input,dim=0).unsqueeze(0)
                        if hparams.cuda:
                            val_input = val_input.cuda()

                        with torch.no_grad():
                            q_values = model(val_input)
                        val_action = eps_greedy(eps=0.05, env=env, q_values=q_values)
                        val_new_state, val_reward, val_done, val_info = env.step(val_action)
                        total_reward += val_reward

                        
                        val_step += 1
                        if val_step > 2500:
                            val_done = True
                            tqdm.write('eval timeout')


                mean_reward = total_reward / hparams.num_eval_episodes
                if mean_reward > best_val:
                    torch.save(model.state_dict(), str(hparams.save_dir / 'dqn.pt'))
                    best_val = mean_reward 
                tqdm.write("Mean val reward: {}".format(mean_reward))
                if hparams.log:
                    wandb.log({'val_reward': mean_reward}, num_evals)
                num_evals += 1

                done = True # reset at beginning of next train iter
                model.train()

            # increment train step counter
            step += 1
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # program args
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--ae_path', type=str, 
            default='checkpoints/autoencoder/20210423_184757/epoch=9.ckpt')
    parser.add_argument('--save_dir', type=Path, default='checkpoints')

    # DRL args
    parser.add_argument('--epoch_len', type=int, default=1000)
    parser.add_argument('--buffer_len', type=int, default=100000)
    parser.add_argument('--history_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--burn_steps', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--visual_freq', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=5)
    parser.add_argument('--rollout_steps_per_iteration', type=int, default=20)
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--k', type=int, default=4)
    args = parser.parse_args()


    args.save_dir = args.save_dir / 'dqn' / datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir.mkdir(parents=True, exist_ok=False)

    main(args)
