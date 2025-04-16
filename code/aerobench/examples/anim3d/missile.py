import numpy as np
class missile():
    """
    Plane model
    Parameters based on AIM120
    source: http://www.designation-systems.net/dusrm/m-120.html
    """

    _g = 9.81  # gravitational acceleration
    # TODO constants
    _Tmax = 4800  # thrust (N)最初是10000
    _S = 0.3  # area (m2)
    _m = 150  # mass (kg)
    _dm = 4  # mass decreasing rate (kg/s)
    _CD = 0.1
    _t_max = 30

    def __init__(self):
        self.m_state = np.array([np.random.uniform(4000,6000),np.random.uniform(4000,6000),np.random.uniform(4000,6000),50, np.deg2rad(135), 0])
        self._t=0

    def step(self, target_state, dt=0.1):
        self._t += dt
        action, info = self._guidance(target_state, dt)
        self._state_trans(action, dt)

    @property
    def _D(self):
        """
        Drag force
        """
        x, y, z, v, phi, theta = self.m_state

        rho0 = 1.225
        rho = rho0 * np.exp(-z / 9300)
        qbar = .5 * rho * np.power(v, 2)

        CD = self._CD
        S = self._S
        D = CD * S * qbar
        return D

    def _guidance(self, state_t, dt):
        """
        Guidance law, proportional navigation
        """
        assert state_t.size == 6, "Only support state shape(C,)"
        x_m, y_m, z_m, v_m, phi_m, theta_m = self.m_state
        x_t, y_t, z_t, v_t, phi_t, theta_t = state_t
        g = self._g
        K = 3  # proportionality constant of proportional navigation
        dx_m, dy_m, dz_m = v_m * np.cos(theta_m) * np.cos(phi_m), v_m * np.cos(theta_m) * np.sin(
            phi_m), v_m * np.sin(theta_m)
        dx_t, dy_t, dz_t = v_t * np.cos(theta_t) * np.cos(phi_t), v_t * np.cos(theta_t) * np.sin(
            phi_t), v_t * np.sin(theta_t)
        Rxy = np.linalg.norm([x_m - x_t,
                              y_m - y_t])  # distance from missile to target project to X-Y plane
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        ny = K * v_m / g * np.cos(theta_m) * dbeta
        nz = K * v_m / g * deps + np.cos(theta_m)
        Rc = 1e3  # available distance
        if Rxyz < Rc:
            info = "Done"
        else:
            info = None
        return np.array([ny, nz]), info

    def _state_trans(self, action, dt):
        """
        State transition function

        Parameters
        ----------
        state: array_like
            State array [x, y, z, v, phi(yaw), theta(pitch)]
        action: array_like
            Action array [ny, nz]
        """
        x, y, z, v, phi, theta = self.m_state
        assert action.size == 2, "Only support action shape (2,)"
        ny, nz = self._trim_action(action)
        g = self._g
        m = self._m
        self._m -= dt * self._dm

        Tmax = self._Tmax
        D = self._D

        nx = (Tmax - D) / (m * g)

        dv = g * (nx - np.sin(theta))
        dphi = g / v * (ny * np.cos(theta))
        dtheta = g / v * (nz - np.cos(theta))

        v += dt * dv
        phi += dt * dphi
        theta += dt * dtheta

        dx = v * np.cos(theta) * np.cos(phi)
        dy = v * np.cos(theta) * np.sin(phi)
        dz = v * np.sin(theta)

        x += dt * dx
        y += dt * dy
        z += dt * dz

        self.m_state = self._trim_state(np.array([x, y, z, v, phi, theta]))

    @staticmethod
    def _trim_action(action):
        n_max = 40  # max overload
        return np.clip(action, -n_max, n_max)

    @staticmethod
    def _trim_state(state):
        state[4:6] = (state[4:6] + np.pi) % (2 * np.pi) - np.pi
        low = np.array([-np.inf, -np.inf, -np.inf, 200, -np.inf, -np.inf])
        high = np.array([np.inf, np.inf, np.inf, 1200, np.inf, np.inf])
        return np.clip(state, low, high)