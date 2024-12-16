"""
waypoint autopilot

ported from matlab v2

用于飞机的自动驾驶系统，专门用于跟踪一组预定的航点

代码主要目的是使飞机能够自动地沿着预定航点路径飞行。它包括了各种用于路径跟踪、高度控制、滚转角控制等的功能

"""

from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np

from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex
from aerobench.lowlevel.low_level_controller import LowLevelController


class WaypointAutopilot(Autopilot):
    """waypoint follower autopilot"""

    def __init__(self, waypoints, gain_str='old', stdout=False):
        """waypoints is a list of 3-tuples"""

        self.stdout = stdout  # 是否向控制台输出信息的标志
        self.waypoints = waypoints  # 航点列表
        self.waypoint_index = 0  # 当前的航点索引

        # waypoint config 航点配置
        self.cfg_slant_range_threshold = 100  # 斜距阈值(即为现在的三维坐标和当前航点三维坐标的mse，判断是否通过该航点)

        # default control when not waypoint tracking 当不进行航点跟踪时的默认控制
        self.cfg_u_ol_default = (0, 0, 0, 0.3)

        # 控制配置
        # 速度控制的增益
        self.cfg_k_vt = 0.25
        self.cfg_airspeed = 550

        # 高度追踪的增益
        self.cfg_k_alt = 0.005
        self.cfg_k_h_dot = 0.02

        # 航向追踪的增益
        self.cfg_k_prop_psi = 5
        self.cfg_k_der_psi = 0.5

        # 滚动追踪的增益
        self.cfg_k_prop_phi = 0.75
        self.cfg_k_der_phi = 0.5
        # self.cfg_max_bank_deg = 75  # 最大倾斜角度设定值
        self.cfg_max_bank_deg = 65  # 最大倾斜角度设定值
        # v2 was 0.5, 0.9

        # Nz的范围
        # self.cfg_max_nz_cmd = 8
        # self.cfg_min_nz_cmd = -2
        self.cfg_max_nz_cmd = 4
        self.cfg_min_nz_cmd = -1

        self.done_time = 0.0  # 完成时间

        # 初始化低级控制器
        llc = LowLevelController(gain_str=gain_str)

        # 使用Autopilot的构造函数初始化该类
        Autopilot.__init__(self, 'Waypoint 1', llc=llc)

    def log(self, s):
        """print to terminal if stdout is true"""

        if self.stdout:
            print(s)

    def get_u_ref(self, _t, x_f16):
        """
        get the reference input signals
        这个方法的主要目标是根据当前飞机状态和下一个航点生成一组参考控制输入，以使飞机能够成功飞到该航点。
        """
        # 如果飞机还没有完成所有航点的飞行
        if self.mode != "Done":
            psi_cmd = self.get_waypoint_data(x_f16)[0]  # 获取下一个航点的目标偏航角（航向）

            # 根据目标航向获取期望的横滚角
            # Get desired roll angle given desired heading
            phi_cmd = self.get_phi_to_track_heading(x_f16, psi_cmd)
            # 获取需要的滚动速度来维持或达到这个横滚角
            ps_cmd = self.track_roll_angle(x_f16, phi_cmd)

            # 获取需要的垂直载荷因子来达到期望的高度
            nz_cmd = self.track_altitude(x_f16)
            # 获取油门指令来达到或维持期望的空速
            throttle = self.track_airspeed(x_f16)
        else:
            # 如果已完成所有航点的飞行：飞平
            # Waypoint Following complete: fly level.
            throttle = self.track_airspeed(x_f16)
            ps_cmd = self.track_roll_angle(x_f16, 0)  # 飞行时保持翼平
            nz_cmd = self.track_altitude_wings_level(x_f16)  # 保持当前高度

        # trim to limits
        # 将命令限制在合理范围内
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        # 保证油门在[0,1]之间
        throttle = max(min(throttle, 1), 0)

        # Create reference vector
        # 创建并返回参考向量
        rv = [nz_cmd, ps_cmd, 0, throttle]

        return rv

    def track_altitude(self, x_f16):
        """
        get nz to track altitude, taking turning into account
        根据不同的高度误差和坡度角，该方法返回一个合适的 nz 值来调整飞机的高度
        """

        h_cmd = self.waypoints[self.waypoint_index][2]

        h = x_f16[StateIndex.ALT]
        phi = x_f16[StateIndex.PHI]

        # Calculate altitude error (positive => below target alt)
        h_error = h_cmd - h
        nz_alt = self.track_altitude_wings_level(x_f16)
        nz_roll = get_nz_for_level_turn_ol(x_f16)

        if h_error > 0:
            # Ascend wings level or banked
            nz = nz_alt + nz_roll
        elif abs(phi) < np.deg2rad(15):
            # Descend wings (close enough to) level
            nz = nz_alt + nz_roll
        else:
            # Descend in bank (no negative Gs)
            nz = max(0, nz_alt + nz_roll)

        return nz

    def get_phi_to_track_heading(self, x_f16, psi_cmd):
        """
        get phi from psi_cmd
        根据当前和目标航向角的差异，以及当前的横滚速度，计算出一个合适的横滚角指令 phi_cmd
        """

        # PD Control on heading angle using phi_cmd as control

        # Pull out important variables for ease of use
        psi = wrap_to_pi(x_f16[StateIndex.PSI])
        r = x_f16[StateIndex.R]

        # Calculate PD control
        psi_err = wrap_to_pi(psi_cmd - psi)

        phi_cmd = psi_err * self.cfg_k_prop_psi - r * self.cfg_k_der_psi

        # Bound to acceptable bank angles:
        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)

        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)

        return phi_cmd

    def track_roll_angle(self, x_f16, phi_cmd):
        """
        get roll angle command (ps_cmd)
        根据当前和目标横滚角的差异，以及当前的横滚速度，计算出一个合适的滚转速率（ps）以实现横滚角的控制
        """

        # PD control on roll angle using stability roll rate

        # Pull out important variables for ease of use
        phi = x_f16[StateIndex.PHI]
        p = x_f16[StateIndex.P]

        # Calculate PD control
        ps = (phi_cmd - phi) * self.cfg_k_prop_phi - p * self.cfg_k_der_phi

        return ps

    def track_airspeed(self, x_f16):
        """get throttle command"""

        vt_cmd = self.cfg_airspeed

        # Proportional control on airspeed using throttle
        throttle = self.cfg_k_vt * (vt_cmd - x_f16[StateIndex.VT])

        return throttle

    def track_altitude_wings_level(self, x_f16):
        """
        get nz to track altitude
        使用PD控制策略和当前的飞行状态信息，来计算一个垂直载荷因子（nz）以使飞机能够跟踪或达到目标高度
        """

        i = self.waypoint_index if self.waypoint_index < len(self.waypoints) else -1

        h_cmd = self.waypoints[i][2]

        vt = x_f16[StateIndex.VT]
        h = x_f16[StateIndex.ALT]

        # Proportional-Derivative Control
        h_error = h_cmd - h
        gamma = get_path_angle(x_f16)
        h_dot = vt * sin(gamma)  # Calculated, not differentiated

        # Calculate Nz command
        nz = self.cfg_k_alt * h_error - self.cfg_k_h_dot * h_dot

        return nz

    def is_finished(self, t, x_f16):
        """is the maneuver done? 判断任务是否完成，并根据当前飞行状态推进离散模式"""

        rv = self.waypoint_index >= len(self.waypoints) and self.done_time+0.5  < t

        return rv

    def advance_discrete_mode(self, t, x_f16):
        """
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        函数最后返回一个布尔值，该值表示离散状态是否已更改
        这个函数主要用于跟踪飞机的航路点，当飞机接近一个航路点时，它会更新航路点索引并记录模式转换。
        函数的返回值rv指示是否发生了模式转换。
        """

        if self.waypoint_index < len(self.waypoints):
            slant_range = self.get_waypoint_data(x_f16)[-1]

            if slant_range < self.cfg_slant_range_threshold:
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.waypoints):
                    self.done_time = t

        premode = self.mode

        if self.waypoint_index >= len(self.waypoints):
            self.mode = 'Done'
        else:
            self.mode = f'Waypoint {self.waypoint_index + 1}'

        rv = premode != self.mode

        #if rv:
            #self.log(f"Waypoint transition {premode} -> {self.mode} at time {t}")

        return rv

    def get_waypoint_data(self, x_f16):
        """returns current waypoint data tuple based on the current waypoint:

        (heading, inclination, horiz_range, vert_range, slant_range)

        heading = heading to tgt, equivalent to psi (rad)
        inclination = polar angle to tgt, equivalent to theta (rad)
        horiz_range = horizontal range to tgt (ft)
        vert_range = vertical range to tgt (ft)
        slant_range = total range to tgt (ft)

        该函数的目的是基于当前航路点，计算并返回一组数据，

        包括航向、倾角、水平距离、垂直距离和总距离。
        """

        waypoint = self.waypoints[self.waypoint_index]

        e_pos = x_f16[StateIndex.POSE]
        n_pos = x_f16[StateIndex.POSN]
        alt = x_f16[StateIndex.ALT]

        delta = [waypoint[i] - [e_pos, n_pos, alt][i] for i in range(3)]

        _, inclination, slant_range = cart2sph(delta)

        heading = wrap_to_pi(pi / 2 - atan2(delta[1], delta[0]))

        horiz_range = np.linalg.norm(delta[0:2])
        vert_range = np.linalg.norm(delta[2])

        return heading, inclination, horiz_range, vert_range, slant_range


def get_nz_for_level_turn_ol(x_f16):
    """get nz to do a level turn 是获取执行水平转弯（level turn）所需的法向过载（nz）"""

    # Pull g's to maintain altitude during bank based on trig

    # Calculate theta
    phi = x_f16[StateIndex.PHI]

    if abs(phi):  # if cos(phi) ~= 0, basically
        nz = 1 / cos(phi) - 1  # Keeps plane at altitude
    else:
        nz = 0

    return nz


def get_path_angle(x_f16):
    """get the path angle gamma 计算飞机的路径角"""

    alpha = x_f16[StateIndex.ALPHA]  # AoA           (rad)
    beta = x_f16[StateIndex.BETA]  # Sideslip      (rad)
    phi = x_f16[StateIndex.PHI]  # Roll anle     (rad)
    theta = x_f16[StateIndex.THETA]  # Pitch angle   (rad)

    gamma = asin((cos(alpha) * sin(theta) - \
                  sin(alpha) * cos(theta) * cos(phi)) * cos(beta) - \
                 (cos(theta) * sin(phi)) * sin(beta))

    return gamma


def wrap_to_pi(psi_rad):
    """handle angle wrapping
    角度标准化
    returns equivelent angle in range [-pi, pi]
    """

    rv = psi_rad % (2 * pi)

    if rv > pi:
        rv -= 2 * pi

    return rv


def cart2sph(pt3d):
    """
    Cartesian to spherical coordinates
    三维的笛卡尔坐标转换为球坐标
    returns az, elev, r
    """

    x, y, z = pt3d

    h = sqrt(x * x + y * y)
    r = sqrt(h * h + z * z)

    elev = atan2(z, h)
    az = atan2(y, x)

    return az, elev, r


if __name__ == '__main__':
    print("Autopulot script not meant to be run directly.")
