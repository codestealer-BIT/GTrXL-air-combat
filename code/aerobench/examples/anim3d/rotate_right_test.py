import math
import matplotlib.pyplot as plt


def rotate_point(dx, dy, psi):
    """
    Rotate a point by a given angle psi.

    :param dx: x component of the point/vector
    :param dy: y component of the point/vector
    :param psi: Rotation angle in radians
    :return: Rotated point as a tuple (new_dx, new_dy)
    """
    new_dx = dx * math.cos(psi) - dy * math.sin(psi)
    new_dy = dx * math.sin(psi) + dy * math.cos(psi)
    return new_dx, new_dy


def calculate_waypoints(x, y, alt, psi):
    waypoints = [[x, y, alt]]

    # Define relative movements
    movements = [[0, 5000], [5000, 6000]]

    for dx, dy in movements:
        # Rotate the movement
        rotated_dx, rotated_dy = rotate_point(dx, dy, psi)

        # Update x and y based on the rotated movement
        x += rotated_dx
        y += rotated_dy

        waypoints.append([x, y, alt])

    return waypoints

# Example usage
x, y, alt = 0, 0, 10000  # Starting point and altitude
psi = math.radians(45)   # Convert 45 degrees to radians for example

# Calculate waypoints
waypoints = calculate_waypoints(x, y, alt, psi)

# For visualization (Optional)
# Extract x and y coordinates for plotting
x_coords = [wp[0] for wp in waypoints]
y_coords = [wp[1] for wp in waypoints]

# Plotting (Optional)
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, marker='o')
plt.title("Flight Path with Psi = 45°")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.axis('equal')  # Ensures equal aspect ratio
plt.show()
