"""
下落
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import numpy as np
from numpy import deg2rad
from aerobench.examples.anim3d.straight_psi import straight_psi
#import matplotlib.pyplot as plt
from aerobench.examples.anim3d.calculate_next_waypoint import calculate_next_waypoint
from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot


def fall_simulate(filename, state,m_state):
    """simulate the system, returning waypoints, res"""
    x=state[10]
    y=state[9]
    z=state[11]
    alt=z
    state[1:5]=[0,0,0,0]
    state[6:9]=[0,0,0]
    init = state
    distance=200
    tmax = 30  # simulation time
    x_next, y_next = calculate_next_waypoint(x, y, state[5], distance)
    waypoints = [[x, y, alt],
                 [x_next, y_next, alt-200]
                 ]
    ap = WaypointAutopilot(waypoints, stdout=True)
    # 环境 + agent
    extended_states = True
    res = run_f16_sim(init, tmax, ap, m_state,extended_states=extended_states, integrator_str='rk45')

    #print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")


    return res
