import math

def calculate_height(angle_degrees, distance, use_hypotenuse=False):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    if use_hypotenuse:
        # height = sin(angle) * hypotenuse
        height = math.sin(angle_radians) * distance
    else:
        # height = tan(angle) * adjacent
        height = math.tan(angle_radians) * distance

    return height

# Example usage
angle = 2  # degrees
distance = 300  # meters

# # Case 1: distance is adjacent side
# height_adjacent = calculate_height(angle, distance, use_hypotenuse=False)
# print(f"Height (adjacent base = {distance} m): {height_adjacent:.2f} m")

# Case 2: distance is hypotenuse
height_hypotenuse = calculate_height(angle, distance, use_hypotenuse=True)
print(f"Height (hypotenuse = {distance} m): {height_hypotenuse:.2f} m")

height_hypotenuse = calculate_height(angle, 100, use_hypotenuse=True)
print(f"Height (hypotenuse = {distance} m): {height_hypotenuse:.2f} m")