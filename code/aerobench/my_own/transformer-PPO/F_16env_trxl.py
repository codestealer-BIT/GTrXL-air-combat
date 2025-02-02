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
import multiprocessing
import random
import torch
from sklearn.preprocessing import MinMaxScaler
from aerobench.examples.anim3d.run_fall import fall_simulate
from aerobench.examples.anim3d.run_rise import rise_simulate
from aerobench.examples.anim3d.run_straight import straight_simulate
from aerobench.examples.anim3d.run_right_turn import right_turn_simulate
from aerobench.examples.anim3d.run_left_turn import left_turn_simulate
from aerobench.examples.anim3d.missile import missile
simulation_functions = [
        fall_simulate,
        left_turn_simulate,
        right_turn_simulate,
        rise_simulate,
        straight_simulate
    ]

dict={"straight_simulate":"直飞",'rise_simulate':'上升','fall_simulate':'下降','left_turn_simulate':'左转','right_turn_simulate':'右转'}
class F_16env_trxl(gym.Env):
    def __init__(self,episodes,filename):
        self.episodes=episodes
        self.filename=filename
        self.success_list=[]
        self.total_count=0
        assert self.total_count<episodes,"总计数不能大于回合总数"
        self.res={}
        self.distance=0
        self.low=np.array([0,0,0],dtype=np.float32)
        self.high=np.array([np.inf,np.inf,np.inf],dtype=np.float32)
        self.success=0
        self.max_count=100
        self.max_episode_steps = 200#要比回合步数大
        self.lock = multiprocessing.Lock()
        self.missile=missile()
    @property
    def observation_space(self):
        """
        Returns:
            {spaces.Box}: The agent observes its current position and the goal locations, which are masked eventually.
        """
        return spaces.Box(low=-np.inf,high=np.inf,shape = (10,), dtype = np.float32)

    @property
    def action_space(self):
        """
        Returns:
            {spaces.Discrete}: The agent has two actions: going left or going right
        """
        return spaces.Discrete(5)


    def seed(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
    
    def calculate_reward(self,step):
        missile=np.array(self.missile.m_state[:3])
        plane=np.array([self.res['states'][-1][10],self.res['states'][-1][9],self.res['states'][-1][11]])
        self.distance= np.linalg.norm(plane-missile)
        if self.distance<5000:
            r_1=-0.01*(5000-self.distance)
        elif self.distance<=10000:#主要集中在这个区间
            r_1=-0.005*(10000-self.distance)
        else:
            r_1=0#过程奖励
        
        r_2 = -5000 if self.distance<=50 else 5000 if self.max_count == step else 0#结果奖励,加一个self，便于判断该step是否取得胜利

        r_3=0
        bound=lambda x: True if 1000<=x<=20000 else False
        if not  bound(self.res['states'][-1][11]):
            r_3=-100
        #cup=lambda x:20*(1.0/(1+np.exp(50*(x-100000)))-1.0/(1+np.exp(50*x))-1)#杯型函数
        #r_3+=cup(self.x)+cup(self.y)+cup(self.z)#边界奖励
        reward=r_1+r_2+r_3
        return reward
    
    
    def calculate_done(self,step):
        done=False if step<self.max_count and self.distance>50 else True
        return done
    
    def step(self,action,step):
        if step==0:
            self._rewards=[]
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action,type(action),)
        select_simulation=simulation_functions[action]
        if step==0:
            self.res=straight_simulate(self.init_state,self.missile)
        else:
            self.res=select_simulation(self.res['states'][-1][:13],self.missile)
        next_state=[self.res['states'][-1][10],self.res['states'][-1][9],self.res['states'][-1][11],self.res['states'][-1][5],self.res['states'][-1][4],
                    self.missile.m_state[0],self.missile.m_state[1],self.missile.m_state[2],self.missile.m_state[4],self.missile.m_state[5]]
        # print(next_state[:3])
        # print(next_state[5],next_state[6],next_state[7])
        # print(self.missile.m_state[0],self.missile.m_state[1],self.missile.m_state[2])
        reward=self.calculate_reward(step)
        self._rewards.append(reward)
        done=self.calculate_done(step)
        if done:
            info = {"success": True if step==self.max_count else False,
                    "reward": sum(self._rewards),
                    "length": len(self._rewards),
                   }#step是100，len是101，因为step对应的worker_current_step+=1是在step函数后面
            if info["success"]==True:
                self.success_list.append(self.total_count)
            self.total_count+=1
            self._rewards=[]
        else:
            info = None
        next_state=MinMaxScaler().fit_transform(np.array(next_state).reshape(-1,1)).reshape(-1)
        return next_state,reward,done,info,self.res
    
    
    def reset(self):
        #[x,y,z,psi,theta,x_m,y_m,z_m,phi_m,theta_m]侧滑phi，俯仰theta，滚转psi（导弹类里定义侧滑是phi）
        #state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]这里的侧滑实则是psi（飞机类里定义侧滑是psi）
        self.init_state=[250,0,0,0,0,np.pi/2,0,0,0,3000,3000,3000,9]
        self.missile._t=0
        self.missile._dm=0.2
        self.missile._m=150
        self.missile.m_state= np.array([np.random.uniform(2000,8000),np.random.uniform(2000,8000),np.random.uniform(2000,8000),50, np.deg2rad(45), 0])
        # self.missile.m_state= np.array([0,0,0,50, np.deg2rad(0),np.deg2rad(45)])
        obs=[self.init_state[10],self.init_state[9],self.init_state[11],self.init_state[5],self.init_state[4],
             self.missile.m_state[0],self.missile.m_state[1],self.missile.m_state[2],self.missile.m_state[4],self.missile.m_state[5]]
        obs=MinMaxScaler().fit_transform(np.array(obs).reshape(-1,1)).reshape(-1)
        return obs                                                                                                                                               