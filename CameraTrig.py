import numpy as np
depth = 50
offset_x = -30
offset_y = 0
tilt=30
C=0
B=0
A=0
## Calculate the l/r, f/b, up/down position of the object relative to the camera
#x = depth * (np.sin(np.radians(offset_x))) # x position (left- and right+)
#y = depth * (np.sin(np.radians(90)) / (np.sin(np.radians(90-offset_x))))
#z = depth * (np.sin(np.radians(offset_y)))
#
#
#print(x)
#print(y)
#print(z)
#
#ASSUMES CAMERA IS LEVEL AND FACING STRAIGHT
#WORKS AS INTENDED,(cartesian world relative) X> left to right, y>backwards to forwards ,z> down to up
#TEST CODE FOR ROT AND TILT
C= -depth * (np.sin(np.radians(tilt + offset_x)))
B= depth * (np.sin(np.radians(90-(tilt + offset_x))))
A= depth * (np.sin(np.radians(offset_y)))

print(A)
print(B)
print(C)
