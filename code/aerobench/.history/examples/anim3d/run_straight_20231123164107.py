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


def straight_simulate(filename, x, y, z, psi, distance):
    """simulate the system, returning waypoints, res"""

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    # deg2rad(2.1215)
    # 攻角上限大概是 25 度
    alpha = 0  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = z  # altitude (ft)
    vt = 300  # initial velocity (ft/sec)
    phi = 0  # Roll angle from wings level (rad) 滚转 横滚
    theta = 0  # Pitch angle from nose level (rad) 俯仰角
    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, y, x, alt, power]
    tmax = 10  # simulation time
    x_next, y_next = calculate_next_waypoint(x, y, psi, distance)
    waypoints = [[x, y, alt],
                 [x_next, y_next, alt]
                 ]
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
