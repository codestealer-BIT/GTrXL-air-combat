import math


def calculate_next_waypoint(x, y, psi, distance):
    # 计算新坐标
    next_x = x + distance * math.sin(psi)
    next_y = y + distance * math.cos(psi)

    return next_x, next_y


