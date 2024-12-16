"""
Stanley Bak

plots 3d animation for 'u_turn' scenario
"""
import math
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot


def simulate(filename, psi, waypoints):
    """simulate the system, returning waypoints, res"""

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    # deg2rad(2.1215)
    # 攻角上限大概是 25 度
    alpha = 0  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    vt = 540  # initial velocity (ft/sec)
    phi = 0  # Roll angle from wings level (rad) 滚转 横滚
    theta = 0  # Pitch angle from nose level (rad) 俯仰角
    # psi = straight_psi()        # Yaw angle from North (rad)
    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    x, y, alt = waypoints[0][0], waypoints[0][1], waypoints[0][2]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, y, x, alt, power]
    tmax = 90  # simulation time

    # python run_u_turn_anim3d.py u_turn.gif
    # D:\AeroBenchVVPython-master\AeroBenchVVPython-master\code\aerobench\examples\anim3d
    # make waypoint list
    ap = WaypointAutopilot(waypoints, stdout=True)
    # 环境 + agent
    step = 1 / 30
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    if filename.endswith('.mp4'):
        skip_override = 4
    elif filename.endswith('.gif'):
        skip_override = 15
    else:
        skip_override = 15

    anim_lines = []
    modes = res['modes']
    modes = modes[0::skip_override]

    def init_extra(ax):
        """initialize plot extra shapes"""

        l1, = ax.plot([], [], [], 'bo', ms=8, lw=0, zorder=50)
        anim_lines.append(l1)

        l2, = ax.plot([], [], [], 'lime', marker='o', ms=8, lw=0, zorder=50)
        anim_lines.append(l2)

        return anim_lines


    return res, init_extra, skip_override, waypoints


def calculate_waypoints(res, dd_x, dd_y, dd_z):
    """
    计算下一时刻的psi和waypoint
    """

    x, y, z = res['states'][-1][10], res['states'][-1][9], res['states'][-1][11]
    distance = math.sqrt((x - dd_x) ** 2 + (y - dd_y) ** 2 + (z - dd_z) ** 2)
    if distance > 10000 and x < 3000:
        waypoints = [
            [x, y, z],
            [x, y + 5000, z]
        ]
        psi = 0
    elif x > 3000:
        waypoints = [
            [x, y, z],
            [x + 5000, y, z]
        ]
        psi = np.pi / 2
    else:
        waypoints = [
            [x, y, z],
            [x, y + 5000, z],
            [x + 5000, y + 11000, z]
        ]
        psi = 0

    return psi, waypoints


def get_dd_info():
    """
    获取导弹信息
    """
    dd_x, dd_y, dd_z = 0, 0, 0
    return dd_x, dd_y, dd_z


def get_done(res):
    """
    设置结束条件
    """
    if res['states'][-1][10] > 10000:
        return True
    else:
        return False


# def main():
#     """main function"""
#
#     filename = ''
#
#     res, init_extra, update_extra, skip_override, waypoints = simulate(filename, psi=0, waypoints=[[0, -5000, 3000],
#                                                                                                    [0, 0, 3000]])
#     while not get_done(res):
#
#         dd_x, dd_y, dd_z = get_dd_info()
#         psi, waypoints = calculate_waypoints(res, dd_x, dd_y, dd_z)
#         res, init_extra, update_extra, skip_override, waypoints = simulate(filename, psi, waypoints)
#
#     anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
#                      elev=27, azim=-107, skip_frames=skip_override,
#                      chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)


def main():
    """main function"""

    filename = ''

    res, init_extra, skip_override, waypoints = simulate(filename, psi=0, waypoints=[[0, -5000, 3000],
                                                                                      [0, 0, 3000]])
    accumulated_res = {
        'status': [], 'times': [], 'states': [], 'modes': [],
        'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': []
    }
    while not get_done(res):

        dd_x, dd_y, dd_z = get_dd_info()
        psi, waypoints = calculate_waypoints(res, dd_x, dd_y, dd_z)
        res, init_extra, skip_override, waypoints = simulate(filename, psi, waypoints)
        accumulated_res['status'] = res['status']
        accumulated_res['times'].extend(res['times'])
        accumulated_res['states'].append(res['states'])
        accumulated_res['modes'].extend(res['modes'])
        if 'xd_list' in res:
            accumulated_res['xd_list'].extend(res['xd_list'])
            accumulated_res['ps_list'].extend(res['ps_list'])
            accumulated_res['Nz_list'].extend(res['Nz_list'])
            accumulated_res['Ny_r_list'].extend(res['Ny_r_list'])
            accumulated_res['u_list'].extend(res['u_list'])
        accumulated_res['runtime'].append(res['runtime'])
    accumulated_res['states'] = np.vstack(accumulated_res['states'])

    anim3d.make_anim(accumulated_res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
                     elev=27, azim=-107, skip_frames=skip_override,
                     chase=True, fixed_floor=True, init_extra=init_extra)


if __name__ == '__main__':
    main()

# 写步骤，1234个动作，每个时间步会随机执行
# 20个时间步
# def main():
#     """main function"""
#
#     if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
#         filename = sys.argv[1]
#         print(f"saving result to '{filename}'")
#     else:
#         filename = ''
#         print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")
#
#     res, init_extra, update_extra, skip_override, waypoints = simulate(filename, psi=0, waypoints=[[0, -5000, 3000],
#                                                                                                    [0, 0, 3000]])
#     while not get_done(res):
#
#         dd_x, dd_y, dd_z = get_dd_info()
#         psi, waypoints = calculate_waypoints(res, dd_x, dd_y, dd_z)
#         res, init_extra, update_extra, skip_override, waypoints = simulate(filename, psi, waypoints)
#
#     anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
#                      elev=27, azim=-107, skip_frames=skip_override,
#                      chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)