'''
Stanley Bak
Autopilot State-Machine Logic
飞机自动驾驶仪

There is a high-level advance_discrete_state() function, which checks if we should change the current discrete state,
and a get_u_ref(f16_state) function, which gets the reference inputs at the current discrete state.
Autopilot 类 : 作为实现不同类型自动驾驶仪的基类
               它需要一个初始模式（一个指定自动机逻辑状态的字符串）和一个可选的低级控制器（LLC）
FixedSpeedAutopilot 类 : 是 Autopilot 的一个子类，用于实现固定速度的自动驾驶

模拟飞机的自动驾驶功能，考虑到了速度、转弯、高度等因素，并允许通过不同的自动驾驶逻辑和限制来进行模拟

'''

import abc
from math import pi

import numpy as np
from numpy import deg2rad

from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.util import Freezable

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, init_mode, llc=None):

        assert isinstance(init_mode, str), 'init_mode should be a string'

        if llc is None:
            # use default
            llc = LowLevelController()

        self.llc = llc
        self.xequil = llc.xequil
        self.uequil = llc.uequil
        
        self.mode = init_mode # discrete state, this should be overwritten by subclasses

        self.freeze_attrs()

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete mode based on the current aircraft state. Returns True iff the discrete mode
        has changed. It's also suggested to update self.mode to the current mode name.
        '''

        return False

    def is_finished(self, t, x_f16):
        '''
        returns True if the simulation should stop (for example, after maneuver completes)

        this is called after advance_discrete_state
        '''

        return False

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals. Override this one
        in subclasses.

        returns four values per aircraft: Nz, ps, Ny_r, throttle
        '''

        return

    def get_checked_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals and check them against ctrl limits
        获取当前离散状态的参考输入信号，并根据控制限制检查它们
        '''

        rv = np.array(self.get_u_ref(t, x_f16), dtype=float)

        assert rv.size % 4 == 0, "get_u_ref should return Nz, ps, Ny_r, throttle for each aircraft"

        # 这一行代码在一个循环中为每个飞机提取四个参考输入信号。
        # 如果只有一个飞机的话，等于直接Nz, _ps, _Ny_r, _throttle = rv[0:4]
        for i in range(rv.size // 4):
            Nz, _ps, _Ny_r, _throttle = rv[4*i:4*(i+1)]

            l, u = self.llc.ctrlLimits.NzMin, self.llc.ctrlLimits.NzMax
            assert l <= Nz <= u, f"autopilot commanded invalid Nz ({Nz}). Not in range [{l}, {u}]"

        return rv

class FixedSpeedAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, p_gain):
        self.setpoint = setpoint
        self.p_gain = p_gain

        init_mode = 'tracking speed'
        Autopilot.__init__(self, init_mode)

    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        x_dif = self.setpoint - x_f16[0]

        return 0, 0, 0, self.p_gain * x_dif
