"""
Stanley Bak
Python Version of F-16 GCAS
ODE derivative code (controlled F16)

计算基于LQR（线性二次调节器）控制的F-16飞机状态的导数（即状态变化率）以及其他相关信息。
用于根据给定的参考输入和当前状态，计算出飞机的下一个状态
"""

from math import sin, cos

import numpy as np
from numpy import deg2rad

from aerobench.lowlevel.subf16_model import subf16_model
from aerobench.lowlevel.low_level_controller import LowLevelController


def controlled_f16(t, x_f16, u_ref, llc, f16_model='morelli', v2_integrators=False):
    """returns the LQR-controlled F-16 state derivatives and more"""

    assert isinstance(x_f16, np.ndarray)
    assert isinstance(llc, LowLevelController)
    assert u_ref.size == 4

    assert f16_model in ['stevens', 'morelli'], 'Unknown F16_model: {}'.format(f16_model)

    # x_ctrl = [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    # u_deg = [throttle command,elevator command,aileron command,rudder command]
    x_ctrl, u_deg = llc.get_u_deg(u_ref, x_f16)

    # Note: Control vector (u) for subF16 is in units of degrees
    xd_model, Nz, Ny, _, _ = subf16_model(x_f16[0:13], u_deg, f16_model)

    # 它检查是否使用名为 v2_integrators 的某种积分器。根据这个值为真还是假，代码将采用两种不同的方法来计算 ps 和 Ny_r
    if v2_integrators:
        # integrators from matlab v2 model
        ps = xd_model[6] * cos(xd_model[1]) + xd_model[8] * sin(xd_model[1])

        Ny_r = Ny + xd_model[8]
    else:
        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

        # Calculate (side force + yaw rate) term
        Ny_r = Ny + x_ctrl[5]

    xd = np.zeros((x_f16.shape[0],))
    xd[:len(xd_model)] = xd_model

    # integrators from low-level controller 从低级控制器获取积分器的导数
    start = len(xd_model)
    end = start + llc.get_num_integrators()
    int_der = llc.get_integrator_derivatives(t, x_f16, u_ref, Nz, ps, Ny_r)
    xd[start:end] = int_der

    # Convert all degree values to radians for output 度到弧度的转换
    u_rad = np.zeros((7,))  # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    u_rad[0] = u_deg[0]  # throttle

    for i in range(1, 4):
        u_rad[i] = deg2rad(u_deg[i])

    u_rad[4:7] = u_ref[0:3]  # inner-loop commands are 4-7

    return xd, u_rad, Nz, ps, Ny_r
