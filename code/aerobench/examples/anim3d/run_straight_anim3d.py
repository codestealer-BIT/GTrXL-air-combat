"""
直飞
"""

import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import numpy as np
from numpy import deg2rad
from straight_psi import straight_psi
import matplotlib.pyplot as plt
from aerobench.examples.anim3d.calculate_next_waypoint import calculate_next_waypoint
from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot


def simulate(filename, x, y, psi, distance):
    """simulate the system, returning waypoints, res"""

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    # deg2rad(2.1215)
    # 攻角上限大概是 25 度
    alpha = 0  # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3000        # altitude (ft)
    vt = 300          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad) 滚转 横滚
    theta = 0       # Pitch angle from nose level (rad) 俯仰角
    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 30  # simulation time
    x_next, y_next = calculate_next_waypoint(x, y, psi, distance)
    waypoints = [[x, y, alt],
                 [x_next, y_next, alt]
                 ]
    ap = WaypointAutopilot(waypoints, stdout=True)
    # 环境 + agent
    step = 1/30
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    if filename.endswith('.mp4'):
        skip_override = 4
    elif filename.endswith('.gif'):
        skip_override = 15
    else:
        skip_override = 30

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

    def update_extra(frame):
        """update plot extra shapes"""

        mode_names = [f'Waypoint {i + 1}' for i in range(len(waypoints))]

        done_xs = []
        done_ys = []
        done_zs = []

        blue_xs = []
        blue_ys = []
        blue_zs = []

        for i, mode_name in enumerate(mode_names):
            if modes[frame] == mode_name:
                blue_xs.append(waypoints[i][0])
                blue_ys.append(waypoints[i][1])
                blue_zs.append(waypoints[i][2])
                break

            done_xs.append(waypoints[i][0])
            done_ys.append(waypoints[i][1])
            done_zs.append(waypoints[i][2])

        anim_lines[0].set_data(blue_xs, blue_ys)
        anim_lines[0].set_3d_properties(blue_zs)

        anim_lines[1].set_data(done_xs, done_ys)
        anim_lines[1].set_3d_properties(done_zs)

    return res, init_extra, update_extra, skip_override, waypoints


def main():
    """main function"""

    x1, y1 = 0, 0
    x2, y2 = 3000, 1000
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res, init_extra, update_extra, skip_override, waypoints = simulate(filename, x1, y1, x2, y2)

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    alt_filename = 'waypoint_altitude.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

    plot.plot_overhead(res, waypoints=waypoints)
    overhead_filename = 'waypoint_overhead.png'
    plt.savefig(overhead_filename)
    print(f"Made {overhead_filename}")
    plt.close()

    anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
                     elev=27, azim=-107, skip_frames=skip_override,
                     chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)


if __name__ == '__main__':

    main()
