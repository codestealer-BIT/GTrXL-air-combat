"""
左转
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


def left_turn_simulate(state,missile):

    x=state[10]
    y=state[9]
    z=state[11]
    alt=z
    init = state
    tmax = 30  # simulation time
    waypoints = calculate_waypoints(x, y, alt, state[5], 'left')

    ap = WaypointAutopilot(waypoints, stdout=True)
    # 环境 + agent
    extended_states = True
    res = run_f16_sim(init, tmax, ap, missile, extended_states=extended_states, integrator_str='rk45')

    #print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")


    return res
