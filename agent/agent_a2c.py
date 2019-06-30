import time

import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from utils.env import MultiEnv, multiActionTransform
from universe.wrappers import Unvectorize
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os
import numpy as np

import pdb

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# random seed
torch.manual_seed(42)
np.random.seed(42)
if use_cuda:
    torch.cuda.manual_seed_all(42)

class AgentA2C:
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.9
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = args.remotes
        self.seed = 42
        self.max_steps = 1e9
        self.grad_norm = 0.5
        self.entropy_weight = 0.05
        self.eps = np.finfo(np.float32).eps.item()

        #######################    NOTE: You need to implement
        self.recurrent = True # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 1000
        self.save_freq = 1
        self.save_dir = './ckpts/'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs =  MultiEnv()
            self.envs.configure(remotes=self.n_processes)

        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        observation = self.envs.reset()
        self.obs_shape = np.transpose(observation[0], (2, 0, 1)).shape
        self.act_shape = args.action_space

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 
        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, 
                eps=1e-5)
        
        if args.test_a2c:
            self.load_model('./ckpts/model_1239.pt')

        self.hidden = None
        self.init_game_setting()
   
    def _update(self):
        # R_t = reward_t + gamma * R_{t+1}
        with torch.no_grad():
            next_value, _, _ = self.model(self.rollouts.obs[-1], 
                                          self.rollouts.hiddens[-1], 
                                          self.rollouts.masks[-1])
        
        self.rollouts.returns[-1] = next_value.detach()
        for step in reversed(range(self.rollouts.rewards.size(0))):
            self.rollouts.returns[step] = self.rollouts.rewards[step] + \
                                            (self.rollouts.returns[step + 1] * \
                                             self.gamma * \
                                             self.rollouts.masks[step + 1])
        
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)
        values, action_probs, _ = self.model(self.rollouts.obs[:-1].view(-1, 
                                                                         self.obs_shape[0], 
                                                                         self.obs_shape[1], 
                                                                         self.obs_shape[2]), 
                                             self.rollouts.hiddens[0], 
                                             self.rollouts.masks[:-1].view(-1, 1))
        distribution = torch.distributions.Categorical(action_probs)
        log_probs = distribution.log_prob(self.rollouts.actions.flatten()).flatten()
        returns = self.rollouts.returns[:-1].flatten()
        values = values.flatten()
        value_loss = F.smooth_l1_loss(returns, values)
        advantages = returns - values
        action_loss = -(log_probs * advantages.detach()).mean()
        entropy = distribution.entropy().mean()
        loss = value_loss + action_loss + (-self.entropy_weight * entropy)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()

        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()

        return loss.item()

    def _step(self, obs, hiddens, masks):
        with torch.no_grad():
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            values, action_probs, hiddens = self.model(obs, hiddens, masks)
            actions = torch.distributions.Categorical(action_probs).sample()
        
        transformed_action = multiActionTransform(actions.cpu().numpy())
        obs, rewards, dones, infos = self.envs.step(transformed_action)
        
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
        obs = torch.from_numpy(obs).to(self.device).permute(0,3,1,2)
        masks = torch.from_numpy(1 - dones).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        penalty_rewards = (1 - masks) * -10
        rewards = rewards + penalty_rewards.double()
        
        self.rollouts.insert(obs, hiddens, actions.unsqueeze(1), values, rewards.unsqueeze(1), masks.unsqueeze(1))
        
    def train(self):
        
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~') 
        
        running_reward = deque(maxlen=self.update_freq*2)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device).permute(0,3,1,2)
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        
        max_reward = 0.0
        counter = 0
        continual_crash = 0
        
        while True:
            try:
                # Update once every n-steps
                for step in range(self.update_freq):
                    self._step(
                        self.rollouts.obs[step],
                        self.rollouts.hiddens[step],
                        self.rollouts.masks[step])

                    # Calculate episode rewards
                    episode_rewards += self.rollouts.rewards[step]
                    for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                        if m == 0:
                            running_reward.append(r.item())
                    episode_rewards *= self.rollouts.masks[step + 1]

                loss = self._update()
                total_steps += self.update_freq * self.n_processes

                # Log & save model
                if len(running_reward) == 0:
                    avg_reward = 0
                else:
                    avg_reward = sum(running_reward) / len(running_reward)

                if total_steps % self.display_freq == 0:
                    print('Steps: %d/%d | Avg reward: %f | Max reward: %f'%
                            (total_steps, self.max_steps, avg_reward, max_reward))
                    with open('a2c_log.txt', 'a') as fout:
                        fout.write(str(avg_reward) + '\n')

                if total_steps % self.save_freq == 0:
                    self.save_model('model_{}.pt'.format(counter), avg_reward)
                    counter += 1

                if avg_reward > max_reward:
                    max_reward = avg_reward
                    self.save_model('model_max_{}.pt'.format(counter), max_reward)
                    counter += 1

                if total_steps >= self.max_steps:
                    break
                    
                continual_crash = 0
                    
            except Exception as e:
                continual_crash += 1

                if continual_crash >= 10:
                    print('============================================================================================================================================')
                    print(e)
                    print("Crashed 10 times -- stopping u suck")
                    print('============================================================================================================================================')
                  
                    raise e
                else:
                    print('#############################################################################################################################################')
                    print(e)
                    print("Env crash, making new env")
                    print('#############################################################################################################################################')

                    time.sleep(60)
                    self.envs = MultiEnv(resize=(250,150))
                    self.envs.configure(remotes=self.n_processes)
                    time.sleep(60)
               

    def save_model(self, filename, max_reward):
        print('model saved: ' + filename + ' (' + str(max_reward) + ')')
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        if use_cuda:
            self.model = torch.load(path)
        else:
            self.model = torch.load(path, map_location='cpu')

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, test=False):
        with torch.no_grad():
            observation = torch.from_numpy(observation).float().permute(0,3,1,2).to(self.device)
            _, action_prob, hidden = self.model(observation, 
                                                self.hidden, 
                                                torch.ones(1, 1).to(self.device))
            self.hidden = hidden
            action = torch.distributions.Categorical(action_prob).sample()
        
        return action.cpu().numpy()
