# We want to design a trajectory along a path, that the drone is able to follow.

# First we need the path.
# Then, we want to design the flat outputs sigma = [x, y, z, \phi]
# and its derivatives, such that the drone is able to follow it (Kumar 2011)

# First, we need z_B. How do we get it?
# z_B = t / ||t||, where t = [x_dotdot, y_dotdot, z_dotdot]
# x_B, y_B can now also be calculated.
# Thus we get the rotation matrix wRb = [x_B, y_B, z_B]

# Kumar also goes on to show, that the angular velocity is also a function of 
# the flat outputs and their derivatives.


