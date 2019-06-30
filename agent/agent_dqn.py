import time

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import pdb

from collections import namedtuple

from agent.agent import Agent
from utils.util import resize
from utils.env import MultiEnv, multiActionTransform

# random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        
        def conv_block(in_filters, out_filters, kernel, stride, bn=True, mp=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, kernel, stride, 1), 
                     nn.LeakyReLU(0.2, inplace=True)]
            
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
                
            if mp:
                block.append(nn.MaxPool2d(kernel, stride))
                
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(channels, 64, 8, 4, mp=False),
            *conv_block(64, 128, 4, 2),
            *conv_block(128, 128, 3, 1, mp=False),
        )
        
        self.fc = nn.Linear(128 * 8 * 14, 512)
        self.head = nn.Linear(512, num_actions)
        self.V = nn.Linear(512, 1) # Dueling
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv_blocks(x))
        
        x = self.lrelu(self.fc(x.view(x.shape[0], -1)))
        v = self.V(x) # Dueling
        q1 = self.head(x)
        q1 = v - q1.mean(1, keepdim=True) + q1
            
        return q1

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.name = args.model_name
        self.env = env
        self.input_channels = args.channels
        self.num_actions = args.action_space
        
        self.memory = ReplayMemory(10000)

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions).to(device)
        self.online_net = DQN(self.input_channels, self.num_actions).to(device)

        if args.test_dqn:
            self.load(self.name)
        
        # discounted reward
        self.GAMMA = 0.995

        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network (default:4)
        self.learning_start = 500 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 1 # frequency to display training progress
        self.save_freq = 10000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network (default:1000)

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if torch.cuda.is_available():
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        # if explore, you randomly samples one action
        # else, use your model to predict action
        if test:
            state = torch.from_numpy(state).to(device).float().permute(0,3,1,2)
            
            with torch.no_grad():
                return self.online_net(state).max(1)[1].view(1,1)[0].cpu().numpy()
        else:
            if sample > eps_threshold:
                state = state.to(device)
                
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.online_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=device)

    def update(self):
        # To update model, we sample some stored experiences as training examples.
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8, device=device)
        next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(tuple(torch.tensor([list(batch.reward)]))).float().to(device)

        # Compute Q(s_t, a) with your model.
        Q_values = self.online_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            next_state_values = torch.zeros(self.batch_size, device=device)
#             next_state_values[mask] = self.target_net(next_states).max(1)[0].detach()
            # Double DQN
            eval_action = self.online_net(next_states).max(1)[1].unsqueeze(1)
            next_state_values[mask] = torch.gather(self.target_net(next_states), dim=1, index=eval_action).squeeze(1)

        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_Q_values = next_state_values * self.GAMMA + reward_batch

        # Compute temporal difference loss
        loss = F.smooth_l1_loss(Q_values, expected_Q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')                              
                                       
        episodes_done_num = 0 # passed episodes
        max_reward = 0
        total_reward = 0 # compute average reward
        loss = 0
        continual_crash = 0

        while(True):
            try:
                state = self.env.reset()
                # State: (150, 250, 3) --> (1, 3, 150, 250)
                state = torch.from_numpy(state).float().permute(0,3,1,2)
                
                done = False
                while(not done):
                    # select and perform action
                    action = self.make_action(state)
                    transformed_action = multiActionTransform(action[0].cpu().numpy())
                    next_state, reward, done, _ = self.env.step(transformed_action)
                    total_reward += reward

                    # process new state
                    next_state = torch.from_numpy(next_state).float().permute(0,3,1,2)
                                           
                    if done:
                        next_state = None

                    # store the transition in memory
                    self.memory.push(state, action, next_state, reward)
                    
                    # move to the next state
                    state = next_state

                    # Perform one step of the optimization
                    if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                        loss = self.update()

                    # update target network
                    if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                        self.target_net.load_state_dict(self.online_net.state_dict())

                    self.steps += 1
                    
                if total_reward > max_reward:
                    max_reward = total_reward    
                    self.save(self.name)
                    
                if self.steps % self.save_freq == 0:   
                    self.save(self.name + '_step')
                
                if episodes_done_num % self.display_freq == 0:
                    print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f | Buffer loads: %f%%'%
                            (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss, (len(self.memory)/self.learning_start)*100))
                    with open(self.name + '_log.txt', 'a') as fout:
                        fout.write(str(total_reward / self.display_freq) + '\n')
                    total_reward = 0

                episodes_done_num += 1
                if self.steps > self.num_timesteps:
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
                    env = MultiEnv()
                    env.configure(remotes=1)
                    self.env = env
                    time.sleep(60)
