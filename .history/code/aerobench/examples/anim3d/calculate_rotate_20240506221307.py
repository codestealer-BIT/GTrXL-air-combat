import math


def rotate_point_left(dx, dy):
    """
    Rotate a point by a given angle psi.

    :param dx: x component of the point/vector
    :param dy: y component of the point/vector
    :param psi: Rotation angle in radians
    :return: Rotated point as a tuple (new_dx, new_dy)
    """
    ANGLE=math.radians(30)
    new_dx = dx * math.cos(ANGLE) - dy * math.sin(ANGLE)
    new_dy = dx * math.sin(ANGLE) + dy * math.cos(ANGLE)
    return new_dx, new_dy

def rotate_point_right(dx, dy):
    """
    Rotate a point by a given angle psi.

    :param dx: x component of the point/vector
    :param dy: y component of the point/vector
    :param psi: Rotation angle in radians
    :return: Rotated point as a tuple (new_dx, new_dy)
    """
    ANGLE=math.radians(30)
    new_dx = dx * math.cos(ANGLE) + dy * math.sin(ANGLE)
    new_dy = -dx * math.sin(ANGLE) + dy * math.cos(ANGLE)
    return new_dx, new_dy

def calculate_waypoints(x, y, alt, psi, direction):
    waypoints = [[x, y, alt]]

    # Define relative movements
    # movements_right = [[0, 5000], [5000, 6000]]
    # movements_left = [[0, 5000], [-5000, 6000]]
    movements = [[2000*math.sin(psi),2000*math.cos(psi)]]

    for dx, dy in movements:
        # Rotate the movement
        if direction=='left':
            rotated_dx, rotated_dy = rotate_point_left(dx, dy)
        elif direction=='right':
            rotated_dx, rotated_dy = rotate_point_right(dx, dy)

        # Update x and y based on the rotated movement
        x += rotated_dx
        y += rotated_dy

        waypoints.append([x, y, alt])

    return waypoints
