import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from agent.agent import Agent
from utils.env import MultiEnv, multiActionTransform
from art import tprint
import pdb

# random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, state_dim, action_num):
        super(Net, self).__init__()

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
            nn.ReLU(),
        )
        self.action = nn.Sequential(
            nn.Linear(64 * 8 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, action_num),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 8 * 14, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        
        action_prob = self.action(x)
        value = self.value(x)  

        return action_prob, value

class AgentAC(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = Net(state_dim = args.channels,
                               action_num= args.action_space)
        self.model = self.model.to(device)
        self.grad_norm = 0.5
        self.entropy_weight = 0.05
        self.args = args
        
        if args.test_ac:
            self.load('ac.cpt')

        # discounted reward
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        
        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 1 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards,  self.prob_value_entropy = [], []
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        # self.model.load_state_dict(torch.load(load_path, map_location='cpu'))
        self.model.load_state_dict(torch.load(load_path))


    def init_game_setting(self):
        self.rewards, self.log_prob, self.entropy, self.value = [], [], [], []

    def make_action(self, state, test=False):
        # Use your model to output distribution over actions and sample from it.
        
        with torch.no_grad():
            state = torch.from_numpy(state).float().permute(0,3,1,2)
            state = state.to(device)

        if test:
            with torch.no_grad():
                action_probs, __ = self.model(state)
                action = torch.distributions.Categorical(action_probs).sample()
                
                return action.cpu().numpy()

        else:
            action_probs, value = self.model(state)

            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
            self.log_prob.append(distribution.log_prob(action))
            self.entropy.append(distribution.entropy())
            self.value.append(value)

            return action.cpu().numpy()

    def update(self):
        # discount your saved reward
        R = 0.0
        returns = []
        for r in self.rewards[::-1]:
            R = int(r) + self.gamma * R
            returns.insert(0,R)
        returns = torch.tensor(returns, device=device)
        action_log_probs = torch.stack(self.log_prob,dim=0)
        self.value = torch.stack(self.value,dim=0)
        entropy = torch.stack(self.entropy,dim=0).mean()
        advantages = returns - self.value
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        loss = value_loss + action_loss - self.entropy_weight * entropy
        # normalize reward
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + self.eps)

        # compute loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.optimizer.step()
        
    def train(self):
        tprint('START')
        max_reward = 0.0
        avg_reward = None # moving average of reward
        continual_crash = 0
        for epoch in range(self.num_episodes):
            try:
                state = self.env.reset()
                self.init_game_setting()
                done = False
                while(not done):
                    if self.args.do_render:
                        self.env.render()

                    action = self.make_action(state)
                    transformed_action = multiActionTransform(action)
                    state, reward, done, _ = self.env.step(transformed_action)

                    self.rewards.append(reward)

                # for logging 
                last_reward = np.sum(self.rewards)
                avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
                with open('ac_log.txt', 'a') as fout:
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
                    self.save('ac.cpt')
                    
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
                    self.env = MultiEnv(resize=(250,150))
                    self.env.configure(remotes=1)
                    time.sleep(60)
