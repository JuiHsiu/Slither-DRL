# Slither-DRL
In this repository, we implement some well known deep reinforcement learning (DRL) algorithms for [slither.io](https://slither.io). 

- Policy Gradient (PG)
- Deep-Q Network (DQN)
- Actor-Critic (AC)
- Advantage Actor-Criitc (A2C)

## Installation Instructions :
- Install [docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for ubuntu 16.04 **MAKE SURE TO DO STEP 2 AS WELL**

- Install [Conda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) for ubuntu 16.04

- Create Conda env
```
conda create --name slither python=3.5
```

- Activate a conda env
```
source activate slither
```

- Install needed packages
```
sudo apt-get update
sudo apt-get install -y tmux htop cmake golang libjpeg-dev libgtk2.0-0 ffmpeg
```

- Install pytorch 1.1.0 for CUDA 10.0
```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl
```

- Install universe installation dependencies
```
pip install numpy
pip install gym==0.9.5
```

- Install universe
```
git clone https://github.com/openai/universe.git
cd universe
pip install -e .
```

- Install this repository
```
cd ..
git clone https://github.com/JuiHsiu/Slither-DRL.git
cd Slither-DRL
```

## How to run :
- training the agent
```
python main.py --train_[pg|dqn|ac|a2c] 
```

- For DQN, there are some improvements can be added: (optional)
```
python main.py --train_dqn [--dueling_dqn] [--prioritized_dqn]
```

- testing the agent
```
python main.py --test_[pg|dqn|ac|a2c]
```

- If you want to see your agent playing the game,
```
python main.py --train_[pg|dqn|ac|a2c] --do_render
```

- By default, when you test the agent, the procedure is recorded as a video. You can assign the directory to the video by :
```
python main.py --test_[pg|dqn|ac|a2c] --video_dir [path_to_save_video]
```

## Advanced arguments
1. Number of Environment

	You can create more than one environment at the same time:
```
	python main.py --train_[pg|dqn|ac|a2c] --remotes [#_of_env]
```
	However, you need to modify the codes to perform batch learning.

2. Action Space

	We make 12 different positions of the mouse as the action space of our agent. If you want the agent to have the ability to accumulate, set the action_space = 24.
```
	python main.py --train_[pg|dqn|ac|a2c] --action_space [12|24]
```

## Demo :
Our best model is A2C and you can see the pre-trained agent playing game as following:

![](demo/a2c.gif)
