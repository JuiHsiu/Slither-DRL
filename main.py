"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
import gym
import universe
import sys
import os

from test import test

from utils.env import MultiEnv
from universe.wrappers import Unvectorize

os.environ['OPENAI_REMOTE_VERBOSE'] = '0'
# os.environ['UNIVERSE_NTPDATE_TIMEOUT'] = '10'

def parse():
    parser = argparse.ArgumentParser(description="Slither.io AI bot")
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_ac', action='store_true', help='whether train Actor-Critic')
    parser.add_argument('--train_a2c', action='store_true', help='whether train A2C')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_ac', action='store_true', help='whether test Actor-Critic')
    parser.add_argument('--test_a2c', action='store_true', help='whether test A2C')
    parser.add_argument('--video_dir', default='records', help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--remotes', type=int, default=1, help='Number of envs.')
    parser.add_argument('--channels', default=3, help='observation input channels')
    parser.add_argument('--action_space', type=int, default=12, help='snake moving action space')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    
    env = MultiEnv(resize=(250,150))
    env.configure(remotes=args.remotes)

    if args.train_pg:
        from agent.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.train()

    if args.train_dqn:
        from agent.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.train()

    if args.train_ac:
        from agent.agent_ac import AgentAC
        agent = AgentAC(env, args)
        agent.train()
        
    if args.train_a2c:
        from agent.agent_a2c import AgentA2C
        agent = AgentA2C(env, args)
        agent.train()

    if args.test_pg:
        from agent.agent_pg import AgentPG
        env = gym.wrappers.Monitor(env, args.video_dir, video_callable=lambda x: True, resume=True)
        agent = AgentPG(env, args)
        test(agent, env, args, total_episodes=1)

    if args.test_dqn:
        from agent.agent_dqn import AgentDQN
        env = gym.wrappers.Monitor(env, args.video_dir, video_callable=lambda x: True, resume=True)
        agent = AgentDQN(env, args)
        test(agent, env, args, total_episodes=1)

    if args.test_ac:
        from agent.agent_ac import AgentAC
        env = gym.wrappers.Monitor(env, args.video_dir, video_callable=lambda x: True, resume=True)
        agent = AgentAC(env, args)
        test(agent, env, args, total_episodes=1)
        
    if args.test_a2c:
        from agent.agent_a2c import AgentA2C
        env = gym.wrappers.Monitor(env, args.video_dir, video_callable=lambda x: True, resume=True)
        agent = AgentA2C(env, args)
        test(agent, env, args, total_episodes=1)

if __name__ == '__main__':
    args = parse()
    run(args)
