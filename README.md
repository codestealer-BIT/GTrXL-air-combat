<p align="center"> <img src="anim3d.gif"/> </p>

Note: This is the v2 branch of the code, which is now a python3 project and includes more modularity and general simulation capabilities. For the original benchmark paper version see the v1 branch.

# AeroBenchVVPython Overview
This project contains a python version of models and controllers that test automated aircraft maneuvers by performing simulations. The hope is to provide a benchmark to motivate better verification and analysis methods, working beyond models based on Dubins car dynamics, towards the sorts of models used in aerospace engineering. Roughly speaking, the dynamics are nonlinear, have about 10-20 dimensions (continuous state variables), and hybrid in the sense of discontinuous ODEs, but not with jumps in the state. 

This is a python port of the original matlab version, which can can see for
more information: https://github.com/pheidlauf/AeroBenchVV

# Citation

For citation purposes, please use: "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, 5th International Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH 2018)

# Required Libraries 

The following Python libraries are required (can be installed using `sudo pip install <library>`):

`numpy` - for matrix operations

`scipy` - for simulation / numerical integration (RK45) and trim condition optimization 

`matplotlib` - for animation / plotting (requires `ffmpeg` for .mp4 output or `imagemagick` for .gif)

`slycot` - for control design (not needed for simulation)

`control` - for control design (not needed for simulation)

### Release Documentation
Distribution A: Approved for Public Release (88ABW-2020-2188) (changes in this version)
    
Distribution A: Approved for Public Release (88ABW-2017-6379) (v1)

run_u_turn_anim3d.py 主要包括设置初始条件、创建航点列表、创建自动驾驶控制器（WaypointAutopilot）、
运行 F-16 模拟（run_f16_sim）并记录模拟结果。
模拟结果包括F-16的状态、时间历史、模式历史等信息。
此外，该函数还包括生成用于动画的初始化和更新函数。进行仿真并将仿真结果输出gif和图,
模拟了飞机从一个初始状态飞向一系列的航点，并在此过程中考虑了飞机的各种物理状态，如速度、姿态角等。
（其中航路点和初始状态变化可以改变飞机的飞行轨迹，合适的航路点和初始状态设置可以得到想要的飞行轨迹，否则会有乱飞的情况）
run_u_turn_anim3d.py中调用的一个重要类叫WaypointAutopilot，这个类继承了另一个名为Autopilot的类，
其作用主要为按照预定的航点来导航飞行器。详细内容可见注释

run_u_turn_anim3d.py中调用的一个重要类叫run_f16_sim
这段代码是用于模拟和分析自主F-16机动的Python函数。以下是代码的主要功能和参数说明：
run_f16_sim 函数用于模拟自主F-16机动。它接受以下参数：
initial_state: 初始状态，描述了F-16飞机的状态。
tmax: 最大模拟时间。
ap: 自动驾驶控制器，用于控制飞机的行为。
step: 积分步长，默认为1/30。
extended_states: 布尔值，如果设置为True，将生成和保存额外的状态信息，如状态导数、控制输入、载荷等。
model_str: 模型字符串，默认为'morelli'。
integrator_str: 积分器字符串，默认为'rk45'，支持的选项为'rk45'和'euler'。
v2_integrators: 布尔值，表示是否使用版本2的积分器。
callback: 回调函数，用于在每个时间步骤后执行自定义操作

函数运行后会返回一个包含以下键值的字典：
'status': 积分状态，应为'finished'（完成）如果没有错误，或者'autopilot finished'（自动驾驶完成）。
'times': 时间历史记录。
'states': 每个时间步骤的状态历史记录。
'modes': 每个时间步骤的模式历史记录。

如果 extended_states 参数为True，则结果还包括以下内容：
'xd_list': 每个时间步骤的导数。
'ps_list': 每个时间步骤的ps值。
'Nz_list': 每个时间步骤的Nz值。
'Ny_r_list': 每个时间步骤的Ny_r值。
'u_list': 每个时间步骤的输入，输入是7元组：throt（油门）、ele（升降舵）、ail（副翼）、rud（方向舵）、Nz_ref、ps_ref、Ny_r_ref

