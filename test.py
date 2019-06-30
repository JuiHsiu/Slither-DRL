import gym
import universe  # register the universe environments
import sys
import argparse
import numpy as np
import pdb

from utils.env import multiActionTransform

seed = 42

def test(agent, env, args, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            if args.do_render:
                env.render()
            transformed_action = multiActionTransform(action)
            state, reward, done, info = env.step(transformed_action)
            episode_reward += reward

        rewards.append(episode_reward)
        
    env.close()
    print('============================================================================================================================================')
    print('============================================================================================================================================')
    print('============================================================================================================================================')       
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    print('============================================================================================================================================')
    print('============================================================================================================================================')
    print('============================================================================================================================================')
