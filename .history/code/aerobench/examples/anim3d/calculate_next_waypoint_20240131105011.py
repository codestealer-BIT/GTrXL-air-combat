import math


def calculate_next_waypoint(x, y, psi, distance):
    # 计算新坐标
    next_x = x + distance * math.cos(psi)
    next_y = y + distance * math.sin(psi)

    return next_x, next_y


