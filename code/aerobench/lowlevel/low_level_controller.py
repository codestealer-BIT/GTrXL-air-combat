'''
Stanley Bak
Low-level flight controller

CtrlLimits类：定义了各种执行机构和飞行参数的控制限制

LowLevelController类： 用于实现飞机的低级控制器，使用了线性二次型调节器（LQR）算法
K_long 和 K_lat：长纵向和横向的 LQR 增益。
xequil 和 uequil：平衡点，可能来自于某种仿真或数学模型。
get_u_deg 函数：根据 F-16 的状态和参考控制输入来计算实际的控制输入。这里也应用了控制限制。
get_num_integrators 函数：获取低级控制器中的积分器数量。
get_integrator_derivatives 函数：获取低级控制器中的积分器导数
'''

import numpy as np
from aerobench.util import Freezable

class CtrlLimits(Freezable):
    """Control Limits"""

    def __init__(self):
        self.ThrottleMax = 1  # Afterburner on for throttle > 0.7
        self.ThrottleMin = 0
        self.ElevatorMaxDeg = 25
        self.ElevatorMinDeg = -25
        self.AileronMaxDeg = 21.5
        self.AileronMinDeg = -21.5
        self.RudderMaxDeg = 30
        self.RudderMinDeg = -30
        
        self.NzMax = 8
        self.NzMin = -3

        self.freeze_attrs()

class LowLevelController(Freezable):
    """
    low level flight controller
    描述一个低级飞行控制器
    """

    old_k_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)
    old_k_lat = np.array([[37.84483, -25.40956, -6.82876, -332.88343, -17.15997],
                          [-23.91233, 5.69968, -21.63431, 64.49490, -88.36203]], dtype=float)

    old_xequil = np.array([502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0, \
                        0.0, 0.0, 0.0, 1000.0, 9.0567], dtype=float).transpose()
    old_uequil = np.array([0.1395, -0.7496, 0.0, 0.0], dtype=float).transpose()

    def __init__(self, gain_str='old'):
        # Hard coded LQR gain matrix from matlab version 获取硬编码的LQR增益矩阵

        assert gain_str == 'old'

        # Longitudinal Gains
        K_long = LowLevelController.old_k_long
        K_lat = LowLevelController.old_k_lat

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_long
        self.K_lqr[1:, 3:] = K_lat

        # equilibrium points from BuildLqrControllers.py 取出均衡点并存储在相应的实例变量中
        self.xequil = LowLevelController.old_xequil
        self.uequil = LowLevelController.old_uequil

        self.ctrlLimits = CtrlLimits()

        self.freeze_attrs()

    def get_u_deg(self, u_ref, f16_state):
        """get the reference commands for the control surfaces 获取控制表面的参考命令"""

        # Calculate perturbation from trim state 计算与均衡状态的偏差
        x_delta = f16_state.copy()
        x_delta[:len(self.xequil)] -= self.xequil

        # Implement LQR Feedback Control 使用LQR增益矩阵计算控制指令
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        # 用于从一个大的数组（x_delta）中选择特定索引的元素，并创建一个新的数组（x_ctrl）
        x_ctrl = np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

        # Initialize control vectors
        u_deg = np.zeros((4,))  # throt, ele, ail, rud

        # Calculate control using LQR gains
        u_deg[1:4] = np.dot(-self.K_lqr, x_ctrl)  # Full Control

        # Set throttle as directed from output of getOuterLoopCtrl(...)
        u_deg[0] = u_ref[3]

        # Add in equilibrium control
        u_deg[0:4] += self.uequil

        # 根据CtrlLimits中定义的限制来限制控制指令的值
        # Limit controls to saturation limits
        ctrlLimits = self.ctrlLimits

        # Limit throttle from 0 to 1
        u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax), ctrlLimits.ThrottleMin)

        # Limit elevator from -25 to 25 deg
        u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg), ctrlLimits.ElevatorMinDeg)

        # Limit aileron from -21.5 to 21.5 deg
        u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg), ctrlLimits.AileronMinDeg)

        # Limit rudder from -30 to 30 deg
        u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg), ctrlLimits.RudderMinDeg)

        return x_ctrl, u_deg

    def get_num_integrators(self):
        """get the number of integrators in the low-level controller 返回低级控制器中的积分器数量"""

        return 3

    def get_integrator_derivatives(self, t, x_f16, u_ref, Nz, ps, Ny_r):
        """get the derivatives of the integrators in the low-level controller 根据给定的输入和状态返回积分器的导数"""

        return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]
