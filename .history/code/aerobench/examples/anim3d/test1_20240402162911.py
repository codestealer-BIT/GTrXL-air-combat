import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import os
os.chdir(os.path.dirname(__file__))
from tqdm import tqdm
import numpy as np
import gym
import collections
import torch
import torch.nn.functional as F
import random
from gym import spaces
from gym.utils import seeding
from aerobench.visualize import anim3d, plot
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


class F_16env(gym.Env):
    def __init__(self):
        self.res={}
        self.missile=[0,0,0]
        self.distance=0
        self.count=0
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        self.init_state=[250,0,0,0,0,np.pi,0,0,0,5000,5000,5000,9]
        self.low=np.array([0,0,0],dtype=np.float32)
        self.high=np.array([np.inf,np.inf,np.inf],dtype=np.float32)
        self.action_space=spaces.Discrete(5)
        self.list=[]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def calculate_reward(self):
        missile=np.array(self.missile)
        plane=np.array([self.x,self.y,self.z])
        self.distance= np.linalg.norm(plane-missile)
        if self.distance<5000:
            reward=-0.05*(5000-self.distance)
        elif self.distance<=10000:#主要集中在这个区间
            reward=-0.05*(10000-self.distance)
        else:
            reward=0
        return reward
    
    def calculate_done(self):
        self.count+=1
        done=False if self.count<30 and self.distance>100 else True
        return done
    
    def step(self,action,m_state):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action,type(action),)
        select_simulation=simulation_functions[action]
        if self.count==0:
            self.res, init_extra, skip_override, _=straight_simulate(filename,self.init_state,m_state)
            self.list.append(dict[straight_simulate.__name__])
        else:
            self.res, init_extra, skip_override, _=select_simulation(filename,self.res['states'][-1][:13],m_state)
            self.list.append(dict[select_simulation.__name__])
        self.x,self.y,self.z=self.res['states'][-1][10],self.res['states'][-1][9],self.res['states'][-1][11]#飞机最新状态回传
        psi,theta=self.res['states'][-1][5],self.res['states'][-1][4]
        x_m,y_m,z_m=self.res['final_state'][:3]
        psi_m,theta_m=self.res['final_state'][4],self.res['final_state'][5]
        next_state=[self.x,self.y,self.z,psi,theta,x_m,y_m,z_m,psi_m,theta_m]
        self.missile=[x_m,y_m,z_m]
        reward=self.calculate_reward()
        done=self.calculate_done()
        return next_state,reward,done,self.res, init_extra, skip_override
    
    
    def reset(self):
        #[x,y,z,psi,theta,x_m,y_m,z_m,psi_m,theta_m]
        return [5000,5000,5000,np.pi,0,0,0,0,np.deg2rad(135),0]



class ReplayBuffer:
    """经验回放池"""
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done
    
    def size(self):
        return len(self.buffer)
    


class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)
    


class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,
                 epsilon,target_update,device):
        self.action_dim=action_dim
        self.q_net=Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        self.target_q_net=Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_update=target_update
        self.count=0
        self.device=device

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            action=self.q_net(state).argmax().item()
        return action
    
    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(
            self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values=self.q_net(states).gather(1,actions)#Q(s,a)
        max_next_q_values=self.target_q_net(next_states).max(1)[0].view(-1,1)#max(1)按行找最大，返回(值，索引)，故取第一个元素
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)
        dqn_loss=torch.mean(F.mse_loss(q_values,q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count%self.target_update==0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count+=1
        
def main():
    lr=2e-3
    hidden_dim=128#隐藏层神经元个数
    gamma=0.98
    epsilon=0.01
    target_update=10
    buffer_size=10000
    minimal_size=20
    batch_size=10
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env=F_16env()
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    state_dim=10
    action_dim=5
    total_iterations=100
    num_episodes = 2000
    replay_buffer=ReplayBuffer(buffer_size)
    separator='-->'
    accumulated_res = {
        'status': [], 'times': [], 'states': [], 'modes': [],'missile':[],'final_state':[],
        'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': []
    }
    agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)
    return_list=[]
    for i in range(total_iterations):
        with tqdm(total=int(num_episodes/total_iterations), desc='Iteration %d' % (i+1)) as pbar:
            for i_episode in range(int(num_episodes/total_iterations)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                env.list=[]
                state = env.reset()
                done=False
                env.count=0#主要是将m_state重新初始化（env.step(action,None if env.count==0 else m_state)）
                while not done:
                    action=agent.take_action(state)
                    next_state,reward,done,res, init_extra, skip_override=env.step(action,None if env.count==0 else m_state)
                    replay_buffer.add(state,action,reward,next_state,done)
                    #print(f"next_state={next_state}\n missile={env.missile}\n{env.distance}")
                    #print(f"reward={reward}\n")
                    m_state=res['final_state']
                    if replay_buffer.size()>minimal_size:
                    n_s,n_a,n_r,n_ns,n_done=replay_buffer.sample(batch_size)
                    transition_dict={
                        'states':n_s,
                        'actions':n_a,
                        'rewards':n_r,
                        'next_states':n_ns,
                        'dones':n_done
                    }
                    state=next_state
                    episode_return += reward
                    if i==total_iterations-1 and i_episode==num_episodes/total_iterations-1:
                        accumulated_res['status'] = res['status']
                        accumulated_res['times'].extend(res['times'])
                        accumulated_res['states'].append(res['states'])
                        accumulated_res['missile'].append(res['missile'])
                        accumulated_res['modes'].extend(res['modes'])
                        accumulated_res['final_state'].append(res['final_state'])
                        if 'xd_list' in res:
                            accumulated_res['xd_list'].extend(res['xd_list'])
                            accumulated_res['ps_list'].extend(res['ps_list'])
                            accumulated_res['Nz_list'].extend(res['Nz_list'])
                            accumulated_res['Ny_r_list'].extend(res['Ny_r_list'])
                            accumulated_res['u_list'].extend(res['u_list'])
                        accumulated_res['runtime'].append(res['runtime'])
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % (num_episodes/total_iterations) == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/total_iterations * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[int(-num_episodes/total_iterations):])})
                pbar.update(1)
            print(separator.join(env.list))#在一个iteration结束后输出动作
    skip_override=10#3
    accumulated_res['states'] = np.vstack(accumulated_res['states'])
    accumulated_res['missile'] = np.vstack(accumulated_res['missile'])
    anim3d.make_anim(accumulated_res, filename, f16_scale=70, viewsize=3000, viewsize_z=4000, trail_pts=np.inf,
                        elev=27, azim=-107, skip_frames=skip_override,
                        chase=True, fixed_floor=False, init_extra=init_extra)
    

if __name__=='__main__':
    main()


