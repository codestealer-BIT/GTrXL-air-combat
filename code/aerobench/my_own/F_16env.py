import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import os
os.chdir(os.path.dirname(__file__))
import gym
import numpy as np
import math
from gym import spaces
from gym.utils import seeding
from aerobench.examples.anim3d.run_fall import fall_simulate
from aerobench.examples.anim3d.run_rise import rise_simulate
from aerobench.examples.anim3d.run_straight import straight_simulate
from aerobench.examples.anim3d.run_right_turn import right_turn_simulate
from aerobench.examples.anim3d.run_left_turn import left_turn_simulate
simulation_functions = [
        fall_simulate,
        left_turn_simulate,
        right_turn_simulate,
        rise_simulate,
        straight_simulate
    ]

dict={"straight_simulate":"直飞",'rise_simulate':'上升','fall_simulate':'下降','left_turn_simulate':'左转','right_turn_simulate':'右转'}
class F_16env(gym.Env):
    def __init__(self,episodes,filename):
        self.filename=filename
        self.success_list=[]
        self.total_count=0
        assert self.total_count<episodes,"总计数不能大于回合总数"
        self.res={}
        self.missile=[0,0,0]
        self.distance=0
        self.count=0
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        self.init_state=[250,0,0,0,0,np.pi/2,0,0,0,3000,3000,3000,9]
        self.low=np.array([0,0,0],dtype=np.float32)
        self.high=np.array([np.inf,np.inf,np.inf],dtype=np.float32)
        self.action_space=spaces.Discrete(5)
        self.list=[[] for _ in range(episodes)]
        self.success=0
        self.max_count=100
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def calculate_reward(self):
        missile=np.array(self.missile)
        plane=np.array([self.x,self.y,self.z])
        self.distance= np.linalg.norm(plane-missile)
        if self.distance<10000:
            r_1=-0.005*(10000-self.distance)
        else:
            r_1=0#过程奖励  
        
        r_2 = -10000 if self.distance<=50 else 10000 if self.max_count == self.count else 0#结果奖励,加一个self，便于判断该step是否取得胜利

        r_3=0
        bound=lambda x: True if 0<=x<=20000 else False
        if not  bound(self.z):
            r_3=-1000
        #cup=lambda x:20*(1.0/(1+np.exp(50*(x-100000)))-1.0/(1+np.exp(50*x))-1)#杯型函数
        #r_3+=cup(self.x)+cup(self.y)+cup(self.z)#边界奖励
        reward=r_1+r_2+r_3
        return reward
    
    def calculate_done(self):
        self.count+=1
        done=False if self.count<self.max_count and self.distance>50 else True
        return done
    
    def step(self,action,m_state):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action,type(action),)
        select_simulation=simulation_functions[action]
        if self.count==0:
            self.res, init_extra, skip_override, _=straight_simulate(self.filename,self.init_state,m_state)
            self.list[self.total_count].append(dict[straight_simulate.__name__])
        else:
            self.res, init_extra, skip_override, _=select_simulation(self.filename,self.res['states'][-1][:13],m_state)
            self.list[self.total_count].append(dict[select_simulation.__name__])
        self.x,self.y,self.z=self.res['states'][-1][10],self.res['states'][-1][9],self.res['states'][-1][11]#飞机最新状态回传
        psi,theta=self.res['states'][-1][5],self.res['states'][-1][4]
        x_m,y_m,z_m=self.res['final_state'][:3]
        psi_m,theta_m=self.res['final_state'][4],self.res['final_state'][5]
        next_state=[self.x,self.y,self.z,psi,theta,x_m,y_m,z_m,psi_m,theta_m]
        self.missile=[x_m,y_m,z_m]
        reward=self.calculate_reward()
        done=self.calculate_done()
        #self.list[self.total_count].append(str(self.distance))
        return next_state,reward,done,self.res, init_extra, skip_override
    
    
    def reset(self):
        #[x,y,z,psi,theta,x_m,y_m,z_m,psi_m,theta_m]
        return [self.init_state[10],self.init_state[9],self.init_state[11],np.pi,0,0,0,0,np.deg2rad(135),0]