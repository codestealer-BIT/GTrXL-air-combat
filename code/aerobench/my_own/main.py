import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import os
os.chdir(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F
import random
from DQN import DQN
from train_off_policy import train_off_policy
from PPO import PPO
from train_on_policy import train_on_policy
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
    filename = sys.argv[1]
    print(f"saving result to '{filename}'")
else:
    filename = ''
    print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")
total_iterations=500#50
num_episodes=5000#500
if __name__=='__main__':
    # agent=PPO(state_dim=10,hidden_dim=128,action_dim=5,actor_lr=1e-3,critic_lr=1e-2,lmbda=0.95,epochs=10,eps=0.2,gamma=0.98)
    # train_on_policy(agent,filename,total_iterations,num_episodes)
    agent=DQN(state_dim=10,hidden_dim=128,action_dim=5,learning_rate=2e-3,gamma=0.98,epsilon=0)
    train_off_policy(agent,filename,total_iterations,num_episodes)


