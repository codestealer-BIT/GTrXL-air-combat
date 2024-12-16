from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import random
from F_16env import F_16env
from buffer import ReplayBuffer
from aerobench.visualize import anim3d, plot
from DQN import DQN

def main():
    minimal_size=200#minimal_size必须大于batch_size
    batch_size=32
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    total_iterations=10
    num_episodes = 200
    env=F_16env(num_episodes,filename)
    env.seed(0)
    replay_buffer=ReplayBuffer(buffer_size=10000)
    accumulated_res = {
        'status': [], 'times': [], 'states': [], 'modes': [],'missile':[],'final_state':[],
        'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': []
    }
    agent=DQN(state_dim=10,hidden_dim=128,action_dim=5,lr=2e-3,gamma=0.98,epsilon=0.01)
    return_list=[]
    for i in range(total_iterations):
        if  env.success_list :
                print('\n\n'.join([f'Trajectory{i+1}:'+str('-->'.join(env.list[index])) for i,index in enumerate(env.success_list)]))#在一个iteration结束后输出动作
                print('\n')
        env.success=0
        env.success_list=[]
        with tqdm(total=int(num_episodes/total_iterations), desc='Iteration %d' % (i+1)) as pbar:
            for i_episode in range(int(num_episodes/total_iterations)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
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
                    state=next_state
                    episode_return += reward
                    if replay_buffer.size()>minimal_size:
                        n_s,n_a,n_r,n_ns,n_done=replay_buffer.sample(batch_size)
                        transition_dict={
                            'states':n_s,
                            'actions':n_a,
                            'rewards':n_r,
                            'next_states':n_ns,
                            'dones':n_done
                        }
                        agent.update(transition_dict)
                    if env.count==env.max_count:
                        env.success+=1
                        env.success_list.append(env.total_count)
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
                env.total_count+=1
                return_list.append(episode_return)
                #if (i_episode+1) % (num_episodes/total_iterations) == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/total_iterations * i + i_episode+1), 'return': '%.2f' % np.mean(return_list[int(-num_episodes/total_iterations):]),'win rate':'%.2f'% (float(env.success)/(num_episodes/total_iterations)*100)+'%'})
                pbar.update(1)
    print('\n\n'.join([f'Trajectory{i+1}:'+str('-->'.join(env.list[index])) for i,index in enumerate(env.success_list)]))#在一个iteration结束后输出动作
    #最后一个iteration结束后输出动作
    skip_override=10#3
    accumulated_res['states'] = np.vstack(accumulated_res['states'])
    accumulated_res['missile'] = np.vstack(accumulated_res['missile'])
    anim3d.make_anim(accumulated_res, filename, f16_scale=70, viewsize=3000, viewsize_z=4000, trail_pts=np.inf,
                        elev=27, azim=-107, skip_frames=skip_override,
                        chase=True, fixed_floor=False, init_extra=init_extra)