import pickle
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from collections import namedtuple
from utils.env import multiActionTransform

from agent_dir.agent import Agent

use_cuda = torch.cuda.is_available()
random.seed(9487)

class DQN(nn.Module):
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(25920, 512)
        self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class DuelingDQN(nn.Module):
    def __init__(self, channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(25920, 512)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

        self.advantage = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage  - advantage.mean()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def penalty(self, ran, num):
        tmp_pos = self.position
        for _ in range(ran):
            tmp_pos = (tmp_pos - 1 + self.capacity) % self.capacity
            self.buffer[tmp_pos]._replace(reward = num)
    
    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.position] = max_prio
        
        # Saves a transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def prioritized_sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 3
        self.num_actions = 12
        # TODO:
        # Initialize your replay buffer
        self.replay_buffer = ReplayBuffer(10000)
        self.prioritized = args.prioritized_dqn

        # build target, online network
        if args.dueling_dqn:
            self.target_net = DuelingDQN(self.input_channels, self.num_actions)
            self.online_net = DuelingDQN(self.input_channels, self.num_actions)
        else:
            self.target_net = DQN(self.input_channels, self.num_actions)
            self.online_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        # save or load model
        self.model_name = 'dqn'
        if args.dueling_dqn:
            self.model_name = 'dueling_' + self.model_name
        if args.prioritized_dqn:
            self.model_name = 'prioritized_' + self.model_name
        if args.test_dqn:
            self.load(self.model_name)
        
        # discounted reward
        self.GAMMA = 0.99
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 1000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 1 # frequency to display training progress
        self.save_freq = 1000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration


    def save_best(self, save_path):
        print('====================================== Save Best Model ======================================')
        print('save best model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online_best.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target_best.cpt')

    def save(self, save_path):
        print('====================================== Save Model ======================================')
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('====================================== Load Model ======================================')
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        if not test:
            # TODO:
            # At first, you decide whether you want to explore the environemnt
            EPS_START, EPS_END, EPS_DECAY = 0.9, 0.01, 20000
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
            is_explore = random.random() <= eps_threshold
            # TODO:
            # if explore, you randomly samples one action
            # else, use your model to predict action
            if is_explore:
                action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
                action = action.cuda() if use_cuda else action
            else:
                with torch.no_grad():
                    action = self.online_net(state).max(1)[1].view(1, 1)
        else:
            state = torch.from_numpy(state).float().permute(0,3,1,2)
            state = state.cuda() if use_cuda else state
            with torch.no_grad():
                    action = self.online_net(state).max(1)[1].view(1, 1)
    
        return action[0].cpu().numpy()

    def update(self):
        # To update model, we sample some stored experiences as training examples.
        if self.prioritized:
            transitions, indices, weights = self.replay_buffer.prioritized_sample(self.batch_size)
        else:
            transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.uint8)
        non_final_mask = non_final_mask.cuda() if use_cuda else non_final_mask
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Compute Q(s_t, a) with your model.
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            next_state_values = torch.zeros(self.batch_size)
            next_state_values = next_state_values.cuda() if use_cuda else next_state_values
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            # DDQN
            #action_index = self.online_net(non_final_next_states).max(1)[1].unsqueeze(1)
            #next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, action_index).squeeze(1)

        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # TODO:
        # Compute temporal difference loss
        if self.prioritized:
            weights = torch.from_numpy(weights).cuda() if use_cuda else torch.from_numpy(weights)
            loss = (state_action_values.squeeze(1) - expected_state_action_values.detach()).pow(2) * weights
            prios = loss + 1e-5
            loss  = loss.mean()
        else:
            l2_loss = nn.MSELoss()
            loss = l2_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        if self.prioritized:
            self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        '''
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        '''
        self.optimizer.step()

        return loss.item()

    def train(self):
        print('====================================== Train ======================================')
        avg_reward_log = []
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        best_reward = 0
        loss = 0 
        while(True):
            state = self.env.reset()
            # State: (batch, height, width, channel) --> (batch, channel, height, width)
            state = torch.from_numpy(state).float().permute(0,3,1,2)
            state = state.cuda() if use_cuda else state
            
            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(multiActionTransform(action))
                reward, done = reward[0], done[0]
                total_reward += reward
                # reward shaping
                reward *= 3
                reward -= 0.1
                if done:
                    self.replay_buffer.penalty(30, -10)
                
                # render
                self.env.render()

                # process new state
                next_state = torch.from_numpy(next_state).float().permute(0,3,1,2)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # store the transition in your replay buffer
                action = torch.from_numpy(action).view(1,1)
                action = action.cuda() if use_cuda else action
                reward = torch.tensor([reward]).cuda() if use_cuda else torch.tensor([reward])
                self.replay_buffer.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save(self.model_name)
                    #self.save_log(avg_reward_log)

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                # save log
                avg_reward = total_reward / self.display_freq
                print('====================================== Log ======================================')
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, avg_reward, loss))
                avg_reward_log.append((self.steps, avg_reward))
                self.save_log(avg_reward_log)
                # save model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_best(self.model_name)
                # reset total reward
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        #self.save(self.model_name)
        self.save_log(avg_reward_log)

    def save_log(self, log):
        print('save log of', self.model_name)
        filename = self.model_name + '.csv'
        with open(filename, 'w') as f:
            for step, reward in log:
                f.write('{}, {}\n'.format(step, reward))