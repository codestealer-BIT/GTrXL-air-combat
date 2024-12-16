import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from run_straight import straight_simulate
class F_16env(gym.Env):
    def __init__(self,init_x,init_y,init_z):
        self.x=init_x
        self.y=init_y
        self.z=init_z
        self.low=np.array([0,0,0],dtype=np.float32)
        self.high=np.array([np.inf,np.inf,np.inf],dtype=np.float32)
        self.action_space=spaces.Discrete(5)
        self.observation_space=spaces.Box(self.low,self.high,dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action,type(action),)
        self.x+=

        
