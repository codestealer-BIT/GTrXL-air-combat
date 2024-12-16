import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class F_16env(gym.Env):
    def __init__(self,init_x,init_y,init_z):
        self.x=init_x
        self.y=init_y
        self.z=init_z
        