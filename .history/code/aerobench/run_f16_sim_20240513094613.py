import time

import numpy as np
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler
from aerobench.examples.anim3d.missile import missile
#step一开始的m_state是None,run_f16_sim里先把missle_sim[-1](即最后一个元素)给res["final_states"],然后在test1的主循环里吧res["final_states"]
#给m_state，然后通过env.step里面的select_simulation把m_state传给run_f16_sim
def run_f16_sim(initial_state, tmax, ap,m_state,step=0.1, extended_states=False, model_str='morelli',
                integrator_str='rk45', v2_integrators=False, callback=None):

    """
    Simulates and analyzes autonomous F-16 maneuvers stevens

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    """
    start = time.perf_counter()
    initial_state = np.array(initial_state, dtype=float)
    llc = ap.llc

    num_vars = len(get_state_names()) + llc.get_num_integrators()

    if initial_state.size < num_vars:
        # append integral error states to state vector
        x0 = np.zeros(num_vars)
        x0[:initial_state.shape[0]] = initial_state
    else:
        x0 = initial_state

    assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

    # run the numerical simulation
    times = [0]
    states = [x0]

    # mode can change at time 0

    ap.advance_discrete_mode(times[-1], states[-1])

    modes = [ap.mode]

    # 如果函数参数extended_states设置为True，还会生成和保存其他信息，例如状态导数、控制输入、载荷等
    if extended_states:
        # 输入t、state、u_ref、低级控制器、机型、积分器种类
        # 得到xd:飞机状态向量的导数。它代表了飞机状态变量的变化率
        # u = [throttle, elevator(弧度), aileron(弧度), rudder(弧度), Nz, ps, Ny_r, throttle]
        xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

        xd_list = [xd]
        u_list = [u]
        Nz_list = [Nz]
        ps_list = [ps]
        Ny_r_list = [Ny_r]

    der_func = make_der_func(ap, model_str, v2_integrators)

    # 选择积分器
    if integrator_str == 'rk45':
        integrator_class = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
    missile_sim=[]
    mis=missile()
    RT_0=np.array([states[-1][10],states[-1][9],states[-1][11]])
    target_state=np.array([*RT_0,states[-1][0],states[-1][5],states[-1][4]])
    if m_state is not None:
        mis.m_state=np.array(m_state).reshape(6,)
    missile_sim.append(mis.m_state)
    flag=False
    while integrator.status == 'running':
        integrator.step()
        if flag==True:
            break
        if integrator.t >= times[-1] + step:
            dense_output = integrator.dense_output()
            while integrator.t >= times[-1] + step:
                t = times[-1] + step
                states.append(dense_output(t))
                #print(f"{round(t, 2)} / {tmax}")
                times.append(t)
                mis.step(target_state)
                target_state=np.array([states[-1][10],states[-1][9],states[-1][11],states[-1][0],states[-1][5],states[-1][4]])
                missile_sim.append(mis.m_state)
                distance=np.linalg.norm(target_state[0:3]-mis.m_state[0:3])
                if distance<50:
                    flag=True
                    #print(f"distance={distance}")
                    break
                #print(f'F16 speed yaw and pitch angle:{states[-1][0],states[-1][5],states[-1][4]}','\n')
                #print(f"missile x y z speed yaw and pitch angle:{mis.m_state[0:]}")
                if callback:
                    callback(states[-1])
                updated = ap.advance_discrete_mode(times[-1], states[-1])
                modes.append(ap.mode)

                # re-run dynamics function at current state to get non-state variables
                if extended_states:
                    xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

                    xd_list.append(xd)
                    u_list.append(u)
                    Nz_list.append(Nz)
                    ps_list.append(ps)
                    Ny_r_list.append(Ny_r)

                if ap.is_finished(times[-1], states[-1]):
                    # this both causes the outer loop to exit and sets res['status'] appropriately
                    integrator.status = 'autopilot finished'
                    break

                if updated:
                    # re-initialize the integration class on discrete mode switches
                    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
                    break

    #assert 'finished' in integrator.status

    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes
    res['missile']= np.array(missile_sim, dtype=float)
    res['final_state']=missile_sim[-1]
    if extended_states:
        res['xd_list'] = xd_list
        res['ps_list'] = ps_list
        res['Nz_list'] = Nz_list
        res['Ny_r_list'] = Ny_r_list
        res['u_list'] = u_list

    res['runtime'] = time.perf_counter() - start

    return res


def make_der_func(ap, model_str, v2_integrators):
    """
    make the combined derivative function for integration
    求组合导数函数进行积分
    """

    def der_func(t, full_state):
        """
        derivative function, generalized for a single aircraft
        微分函数
        """
        

        # 输入时间和full_state 得到rv = Nz, ps, Ny_r, throttle

        u_refs = ap.get_checked_u_ref(t, full_state)
        # assert len(u_refs) == 4, "Unexpected number of control inputs"

        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        # assert len(full_state) == num_vars, "Unexpected state size for a single aircraft"

        state = full_state[:num_vars]
        u_ref = u_refs[:4]

        # 导数状态 xd
        rv = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]

        return rv

    return der_func


# def make_der_func(ap, model_str, v2_integrators):
#     """make the combined derivative function for integration"""
#
#     def der_func(t, full_state):
#         """derivative function, generalized for multiple aircraft"""
#
#         # 输入时间和full_state 得到rv = Nz, ps, Ny_r, throttle
#         u_refs = ap.get_checked_u_ref(t, full_state)
#
#         num_aircraft = u_refs.size // 4
#         num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
#         assert full_state.size // num_vars == num_aircraft
#
#         xds = []
#         # 允许多架飞机输入
#         for i in range(num_aircraft):
#             state = full_state[num_vars*i:num_vars*(i+1)]
#             u_ref = u_refs[4*i:4*(i+1)]
#
#             # 导数状态 xd
#             xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
#             xds.append(xd)
#
#         rv = np.hstack(xds)
#
#         return rv
#
#     return der_func


def get_extended_states(ap, t, full_state, model_str, v2_integrators):

    """
    get xd, u, Nz, ps, Ny_r at the current time / state
    returns tuples if more than one aircraft
    """

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    # 输入时间和full_state 得到rv = Nz, ps, Ny_r, throttle
    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        # 输入t、state、u_ref、低级控制器、机型、积分器种类
        # 得到xd:飞机状态向量的导数。它代表了飞机状态变量的变化率
        # u = [throttle, elevator(弧度), aileron(弧度), rudder(弧度), Nz, ps, Ny_r, throttle]
        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r
