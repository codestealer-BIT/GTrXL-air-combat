from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from F_16env import F_16env
from buffer import ReplayBuffer
from aerobench.visualize import anim3d, plot
from plot import plot
def train_on_policy(agent,filename,total_iterations,num_episodes):
    return_list = []
    env=F_16env(num_episodes,filename)
    env.seed(0)
    accumulated_res = {
        'status': [], 'times': [], 'states': [], 'modes': [],'missile':[],'final_state':[],
        'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': []
    }
    return_list=[]
    win_rate_list=[]
    return_list_big=[]
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
                done = False
                env.count=0
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done,res, init_extra, skip_override=env.step(action,None if env.count==0 else m_state)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    m_state=res['final_state']
                    state = next_state
                    episode_return += reward
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
                agent.update(transition_dict)
                #if (i_episode+1) % (num_episodes/total_iterations) == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/total_iterations * i + i_episode+1), 'return': '%.2f' % np.mean(return_list[int(-num_episodes/total_iterations):]),'win rate':'%.2f'% (float(env.success)/(num_episodes/total_iterations)*100)+'%'})
                pbar.update(1)
            win_rate_list.append((float(env.success)/(num_episodes/total_iterations)*100))
            return_list_big.append(np.mean(return_list[int(-num_episodes/total_iterations):]))
    print('\n\n'.join([f'Trajectory{i+1}:'+str('-->'.join(env.list[index])) for i,index in enumerate(env.success_list)]))#在一个iteration结束后输出动作
    #最后一个iteration结束后输出动作
    print('\n',win_rate_list)
    print('\n',return_list_big)
    skip_override=10#3
    accumulated_res['states'] = np.vstack(accumulated_res['states'])
    accumulated_res['missile'] = np.vstack(accumulated_res['missile'])
    plot(return_list,'F_16env','Episodes','Returns',num_episodes/total_iterations-1)#window_size必须为奇数
    plot(win_rate_list,'F_16env','Iterations','Win_rate(%)',num_episodes/total_iterations-1)
    anim3d.make_anim(accumulated_res, filename, f16_scale=70, viewsize=30000, viewsize_z=4000, trail_pts=np.inf,
                        elev=27, azim=-107, skip_frames=skip_override,
                        chase=True, fixed_floor=False, init_extra=init_extra)