import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.agent import Agent
from utils.util import resize
from utils.env import createEnv, actionTransform
from universe.wrappers import Unvectorize

import pdb

# random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(PolicyNet, self).__init__()

        def conv_block(in_filters, out_filters, kernel, stride, bn=True, mp=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, kernel, stride, 1), 
                     nn.LeakyReLU(0.2, inplace=True)]
            
            if mp:
                block.append(nn.MaxPool2d(kernel, stride))
            
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
                
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(state_dim, 32, 8, 4, mp=False),
            *conv_block(32, 64, 4, 2),
            *conv_block(64, 64, 3, 1, mp=False),
        )

        self.fc1 = nn.Linear(64 * 8 * 14, 512)
        self.fc2 = nn.Linear(512, action_num)

    def forward(self, x):
        x = F.relu(self.conv_blocks(x))
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        action_prob = F.softmax(x, dim=1)
        
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = args.channels,
                               action_num= args.action_space)
        self.model = self.model.to(device)

        self.args = args
        
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        
        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 1 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions, self.saved_log_probs = [], [], []
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path, map_location='cpu'))

    def init_game_setting(self):
        self.rewards, self.saved_actions, self.saved_log_probs = [], [], []

    def make_action(self, state, test=False):
        # Use your model to output distribution over actions and sample from it.
        
        with torch.no_grad():
            state = torch.from_numpy(state).float().permute(2,0,1).unsqueeze(0)
            state = state.to(device)

        if test:
            with torch.no_grad():
                action_probs = self.model(state)
                action = torch.distributions.Categorical(action_probs).sample()
                
                return action.item()

        else:
            action_probs = self.model(state)

            distribution = torch.distributions.Categorical(action_probs)

            action = distribution.sample()
            self.saved_log_probs.append(distribution.log_prob(action))

            return action.item()

    def update(self):
        # discount your saved reward
        R = 0.0
        discounted_rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.append(R)
        discounted_rewards = torch.tensor(discounted_rewards[::-1], device=device)

        # normalize reward
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + self.eps)

        # compute loss 
        loss = 0.0
        for log_prob, R in zip(self.saved_log_probs, discounted_rewards):
            loss += (-log_prob * R)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        max_reward = 0.0
        avg_reward = None # moving average of reward
        continual_crash = 0
        for epoch in range(self.num_episodes):

            try:
                state = self.env.reset()
                state = resize(state)
                self.init_game_setting()
                done = False
                while(not done):
                    if self.args.do_render:
                        self.env.render()

                    action = self.make_action(state)
                    transformed_action = actionTransform(action)
                    state, reward, done, _ = self.env.step(transformed_action)
                    state = resize(state)

                    self.saved_actions.append(action)
                    self.rewards.append(reward)

                # for logging 
                last_reward = np.sum(self.rewards)
                avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
                with open('pg_log.txt', 'a') as fout:
                    fout.write(str(avg_reward) + '\n')
                
                # update model
                self.update()

                if epoch % self.display_freq == 0:
                    print('============================================================================================================================================')
                    print('Epochs: %d/%d | Avg reward: %f '%
                           (epoch, self.num_episodes, avg_reward))
                    print('============================================================================================================================================')
                    
                
                if avg_reward > max_reward:
                    max_reward = avg_reward
                    self.save('pg.cpt')
                    
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
                    env = createEnv()
                    env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'subsample_level': 0, 'encoding': 'tight', 'compress_level': 2, 'fine_quality_level': 100})
                    self.env = env
                    time.sleep(60)
