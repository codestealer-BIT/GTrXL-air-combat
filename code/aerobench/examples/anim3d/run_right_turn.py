"""
右转
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import numpy as np
from numpy import deg2rad
from aerobench.examples.anim3d.straight_psi import straight_psi
import matplotlib.pyplot as plt
from aerobench.examples.anim3d.calculate_rotate import calculate_waypoints
from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot


def right_turn_simulate(filename,state,m_state):
    """simulate the system, returning waypoints, res"""

    x=state[10]
    y=state[9]
    z=state[11]
    alt=z
    init = state
    tmax = 30  # simulation time
    waypoints = calculate_waypoints(x, y, alt, state[5], 'right')

    ap = WaypointAutopilot(waypoints, stdout=True)
    # 环境 + agent

    extended_states = True
    res = run_f16_sim(init, tmax, ap,m_state,extended_states=extended_states, integrator_str='rk45')

    #print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

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

    return res, None, skip_override, waypoints
