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


def fall_simulate(state,missile):
    """simulate the system, returning waypoints, res"""
    x=state[10]
    y=state[9]
    z=state[11]
    alt=z
    state[1:5]=[0,0,0,0]
    state[6:9]=[0,0,0]
    init = state
    distance=2000
    tmax = 30  # simulation time
    x_next, y_next = calculate_next_waypoint(x, y, state[5], distance)
    waypoints = [
                 [x_next, y_next, alt-300]
                 ]
    ap = WaypointAutopilot(waypoints, stdout=True)
    # 环境 + agent
    extended_states = True
    res = run_f16_sim(init, tmax, ap, missile,extended_states=extended_states, integrator_str='rk45')
   #  distance=300
   #  x,y,z,psi=res["states"][-1][10],res["states"][-1][9],res["states"][-1][11],res["states"][-1][5]
   #  x_next, y_next = calculate_next_waypoint(x, y, psi, distance)
   #  waypoints = [[x, y, z],
   #               [x_next, y_next, z]
   #               ]
   #  ap = WaypointAutopilot(waypoints, stdout=True)
   #  res1=run_f16_sim(res["states"][-1][:13],tmax,ap,missile,extended_states=extended_states,integrator_str='rk45')
   #  res = {k: np.concatenate([res[k], res1[k]], axis=0) if isinstance(res[k], np.ndarray) 
   #     else [np.concatenate([res[k][0], res1[k][0]], axis=0)] if isinstance(res[k], list) and isinstance(res[k][0], np.ndarray) 
   #     else res[k] + res1[k] for k in res }
    #print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")


    return res
