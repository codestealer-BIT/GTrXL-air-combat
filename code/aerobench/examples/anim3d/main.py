import math
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
from aerobench.examples.anim3d.run_fall import fall_simulate
from aerobench.examples.anim3d.run_rise import rise_simulate
from aerobench.examples.anim3d.run_straight import straight_simulate
from aerobench.examples.anim3d.run_right_turn import right_turn_simulate
from aerobench.examples.anim3d.run_left_turn import left_turn_simulate

import random

def main():
    """main function"""

    # filename = ''
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
    init_state=[300,0,0,0,0,np.pi,0,0,0,10000,10000,10000,9]
    accumulated_res = {
        'status': [], 'times': [], 'states': [], 'modes': [],
        'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': [],'missile':[]
    }
    n = 0
    while n < 100:
        if n == 0:
            res, init_extra, skip_override, waypoints =straight_simulate(filename,
                                                                          init_state,m_state=np.array([np.nan]*6))
        else :
            selected_simulation = random.choice(simulation_functions)
            #print(selected_simulation)
            res, init_extra, skip_override, waypoints =fall_simulate(filename,
                                                                          res['states'][-1][:13],m_state=np.array([np.nan]*6))

        accumulated_res['status'] = res['status']
        accumulated_res['times'].extend(res['times'])
        accumulated_res['states'].append(res['states'])
        accumulated_res['modes'].extend(res['modes'])
        accumulated_res['missile'].append(res['missile'])
        if 'xd_list' in res:
            accumulated_res['xd_list'].extend(res['xd_list'])
            accumulated_res['ps_list'].extend(res['ps_list'])
            accumulated_res['Nz_list'].extend(res['Nz_list'])
            accumulated_res['Ny_r_list'].extend(res['Ny_r_list'])
            accumulated_res['u_list'].extend(res['u_list'])
        accumulated_res['runtime'].append(res['runtime'])
        n += 1
    accumulated_res['states'] = np.vstack(accumulated_res['states'])
    accumulated_res['missile'] = np.vstack(accumulated_res['missile'])
    anim3d.make_anim(accumulated_res, filename, f16_scale=70, viewsize=30000, viewsize_z=4000, trail_pts=np.inf,
                     elev=27, azim=-107, skip_frames=skip_override,
                     chase=True, fixed_floor=False, init_extra=None)


if __name__ == '__main__':
    main()
# def main():
#     filename = ''
#     res, init_extra, skip_override, waypoints = straight_simulate(filename, 0, 0, 3000, 0, 3000)
#
#     # 定义所有可能的仿真动作函数
#     simulation_functions = [
#         fall_simulate,
#         left_turn_simulate,
#         right_turn_simulate,
#         rise_simulate,
#         straight_simulate
#     ]
#
#     # 初始化累积结果的数据结构
#     accumulated_res = {'times': [], 'states': [], 'modes': [], 'xd_list': [], 'ps_list': [], 'Nz_list': [],
#                        'Ny_r_list': [], 'u_list': [], 'runtime': 0, 'status': res['status']}
#
#     # 随机选择仿真动作20次
#     for i in range(5):
#         selected_simulation = random.choice(simulation_functions)
#         psi = accumulated_res['states'][i][-1][5]
#         # distance = random.randint(1000, 4000)
#         distance = 1000
#         # 执行仿真
#         res, init_extra, skip_override, waypoints = selected_simulation(filename, accumulated_res['states'][i][-1][10],
#                                                                         accumulated_res['states'][i][-1][9],
#                                                                         accumulated_res['states'][i][-1][11],
#                                                                         psi, distance)
#
#         accumulated_res['status'] = res['status']
#         accumulated_res['times'].extend(res['times'])
#         accumulated_res['states'].append(res['states'])
#         accumulated_res['modes'].extend(res['modes'])
#         if 'xd_list' in res:
#             accumulated_res['xd_list'].extend(res['xd_list'])
#             accumulated_res['ps_list'].extend(res['ps_list'])
#             accumulated_res['Nz_list'].extend(res['Nz_list'])
#             accumulated_res['Ny_r_list'].extend(res['Ny_r_list'])
#             accumulated_res['u_list'].extend(res['u_list'])
#         accumulated_res['runtime'] += (res['runtime'])
#     accumulated_res['states'] = np.vstack(accumulated_res['states'])
#
#     # 创建动画
#     anim3d.make_anim(accumulated_res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
#               elev=27, azim=-107, skip_frames=skip_override, chase=True, fixed_floor=True)
#
#
# if __name__ == "__main__":
#     main()
