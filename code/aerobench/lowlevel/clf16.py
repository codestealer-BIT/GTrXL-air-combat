'''
Stanley Bak
clf16.py for F-16 model

This is the objective function for finding the trim condition of the initial states

用于计算F-16飞机的修剪条件的目标函数
修剪条件是飞机在平衡状态下的操作条件，其中所有动力学变量（如速度、角度、控制输入等）都是常数。

'''

from math import asin, sin

from tgear import tgear
from conf16 import conf16
from subf16_model import subf16_model

def clf16(s, x, u, const, model='stevens', adjust_cy=True):
    '''
    objective function of the optimization to find the trim conditions

    x and u get modified in-place
    returns the cost

    '''

    _, singam, _, _, tr, _, _, _, thetadot, _, _, orient = const
    gamm = asin(singam)

    if len(s) == 3:
        u[0] = s[0]
        u[1] = s[1]
        x[1] = s[2]
    else:
        u[0] = s[0]
        u[1] = s[1]
        u[2] = s[2]
        u[3] = s[3]
        x[1] = s[4]
        x[3] = s[5]
        x[4] = s[6]

    #
    # Get the current power and constraints
    #
    x[12] = tgear(u[0])
    [x, u] = conf16(x, u, const)

    # we just want the derivative
    subf16 = lambda x, u: subf16_model(x, u, model, adjust_cy)[0]

    xd = subf16(x, u)

    #
    # Steady Level flight
    #
    if orient == 1:
        r = 100.0*(xd[0]**2 + xd[1]**2 + xd[2]**2 + xd[6]**2 + xd[7]**2 + xd[8]**2)

    #
    # Steady Climb
    #
    if orient == 2:
        r = 500.0*(xd[11]-x[0]*sin(gamm))**2 + xd[0]**2 + 100.0*(xd[1]**2 + xd[2]**2) + \
            10.0*(xd[6]**2 + xd[7]**2 + xd[8]**2)



    #
    # Coord Turn
    #
    if orient == 3:
        r = xd[0]*xd[0] + 100.0 * (xd[1] * xd[1] + xd[2]*xd[2] + xd[11]*xd[11]) + 10.0*(xd[6]*xd[6] + \
            xd[7]*xd[7]+xd[8]*xd[8]) + 500.0*(xd[5] - tr)**2

    #
    # Pitch Pull Up
    #

    if orient == 4:
        r = 500.0*(xd[4]-thetadot)**2 + xd[0]**2 + 100.0*(xd[1]**2 + xd[2]**2) + 10.0*(xd[6]**2 + xd[7]**2 + xd[8]**2)

    #
    # Scale r if it is less than 1
    #
    if r < 1.0:
        r = r**0.5

    return r

# 参数
# s: 包含控制输入和状态变量的向量，用于优化。
# x: 当前状态变量向量。
# u: 当前控制输入向量。
# const: 包含飞机和飞行条件的常量。
# model: 用于计算飞机动力学的模型，默认为"stevens"。
# adjust_cy: 一个布尔值，用于调整侧力系数，默认为True。
# 输出
# 返回一个成本函数r，该函数用于量化当前状态和控制输入的优化程度。
# 主要步骤
# 初始化: 根据const和输入向量s初始化状态x和控制输入u。
# 更新飞机状态和控制输入: 调用conf16函数来获得修正后的状态和控制输入。
# 计算状态导数: 调用subf16_model来获得状态导数xd。
# 计算成本函数: 根据当前的飞行方向（由orient决定）和状态导数来计算成本函数r。
# 对于水平飞行（orient == 1），成本主要依赖于状态导数的平方和。
# 对于爬升飞行（orient == 2），成本包括垂直速度和状态导数。
# 对于配合转弯（orient == 3），成本包括转弯半径和状态导数。
# 对于俯仰拉起（orient == 4），成本包括俯仰角速度和状态导数。
# 返回成本函数: 如果成本函数值小于1，则对其进行平方根处理，然后返回。
# 这个函数通常用于与优化算法（如梯度下降、遗传算法等）结合使用，以找到使成本函数最小化的状态和控制输入。这有助于确定飞机在不同飞行状态下的最优操作条件。