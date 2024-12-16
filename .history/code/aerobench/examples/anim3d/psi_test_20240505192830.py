import math


def degrees_to_radians(deg):
    """
    Convert an angle from degrees to radians.
    角度到弧度

    :param deg: Angle in degrees
    :return: Angle in radians
    """
    return math.radians(deg)


def radians_to_degrees(rad):
    """
    Convert an angle from radians to degrees.
    弧度到角度

    :param rad: Angle in radians
    :return: Angle in degrees
    """
    return math.degrees(rad)


# print(degrees_to_radians(45))
# print(radians_to_degrees(1.56))