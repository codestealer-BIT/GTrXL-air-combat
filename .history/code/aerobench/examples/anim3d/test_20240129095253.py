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


def main():
    filename = ''
    res, init_extra, skip_override, waypoints = rise_simulate(filename, 0, 0, 3000, 0, 2000)
    # 创建动画
    anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
              elev=27, azim=-107, skip_frames=skip_override, chase=True, fixed_floor=True)


if __name__ == "__main__":
    main()
