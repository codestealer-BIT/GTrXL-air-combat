import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from aerobench.examples.anim3d.run_fall import fall_simulate
from aerobench.examples.anim3d.run_rise import rise_simulate
from aerobench.examples.anim3d.run_straight import straight_simulate
from aerobench.examples.anim3d.run_right_turn import right_turn_simulate
from aerobench.examples.anim3d.run_left_turn import left_turn_simulate


if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
    filename = sys.argv[1]
    print(f"saving result to '{filename}'")
else:
    filename = ''
    print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")
simulation_functions = [
        fall_simulate,
        left_turn_simulate,
        right_turn_simulate,
        rise_simulate,
        straight_simulate
    ]
filename=''
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
        select_simulation=simulation_functions[action]
        self.x,self.y,self.z=select_simulation()

        
