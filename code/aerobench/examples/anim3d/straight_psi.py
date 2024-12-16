import math
import numpy


def straight_psi(x1, y1, x2, y2):
    """
    Calculate the heading angle (psi) of a straight line between two points.
    The angle is measured counterclockwise from the positive y-axis.

    :param x1: x-coordinate of the starting point
    :param y1: y-coordinate of the starting point
    :param x2: x-coordinate of the ending point
    :param y2: y-coordinate of the ending point
    :return: Heading angle (psi) in radians
    """

    delta_x = x2 - x1
    delta_y = y2 - y1

    psi = math.atan2(delta_x, delta_y)
    return psi


x1, y1 = 0, 0
x2, y2 = 3999.7668809198653, 43.184468233069566
# print(straight_psi(x1, y1, x2, y2))
