-


# Assume these are the Euler angles and translation vector of the camera RPY
alpha, beta, gamma = np.radians(0), np.radians(90), np.radians(0)  # Convert to radians
#In the above code, alpha, beta, and gamma are the rotation angles of the camera relative to the robot.
#They represent the rotation around the x-axis (roll), y-axis (pitch), and z-axis (yaw) respectively. 
# These angles are usually given in degrees, but the np.radians() function is used to convert them to radians because the R.from_euler() function requires radians


#!!I think its pitch, Roll, left to right
cx, cy, cz = 0, 0, 0
#cx, cy, and cz represent the position of the camera relative to the robot. 
# cx is the distance along the x-axis, cy is the distance along the y-axis, and cz is the distance along the z-axis.

# Create rotation matrix using scipy
r = R.from_euler('xyz', [alpha, beta, gamma])
rotation_matrix = r.as_matrix()

# Create translation vector
translation_vector = np.array([cx, cy, cz]).reshape((3, 1))

# Create transformation matrix
transformation_matrix = np.hstack((rotation_matrix, translation_vector))
transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # Add row for homogeneous coordinates

# Assume these are the object coordinates in the camera's coordinate system
#x, y, z = 40, 50, 60

# Convert object coordinates to homogeneous coordinates
object_coords = np.array([x, y, z, 1]).reshape((4, 1))

# Transform object coordinates to robot coordinates
robot_coords = np.dot(transformation_matrix, object_coords)
robot_coords = robot_coords[:3] / robot_coords[3]  # Convert back from homogeneous coordinates

print(robot_coords)

#The x, y, and z coordinates follow the right-hand rule convention:

#The x-coordinate represents the horizontal displacement of the object to the right (positive) or left (negative) of the camera.
#The y-coordinate represents the depth or distance of the object in front of (positive) or behind (negative) the camera.
#The z-coordinate represents the vertical displacement of the object above (positive) or below (negative) the camera.