import numpy as np
from pyquaternion import Quaternion


def quat_rotation(quat, point):
    point = np.array([0] + point)
    p0, p1, p2, p3 = quat[0], quat[1], quat[2], quat[3]
    quat_conjugate = np.array([p0, -p1, -p2, -p3])
    
    # p (quat_product) q = Q(p) * q
    # Let's define Q
    def get_Q(p):
        p0, p1, p2, p3 = quat[0], quat[1], quat[2], quat[3]
        Q = [[p0, -p1, -p2, -p3],
             [p1, p0, -p3, p2],
             [p2, p3, p0, -p1],
             [p3, -p2, p1, p0]]
        return np.array(Q)
    
    tmp = np.dot(get_Q(quat), point)
    return np.dot(get_Q(tmp), quat_conjugate)


def normalize(v, tolerance=0.00001):
    # https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v
def q_mult(q1, q2):
    # https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    # if x1 == z1 == w2 == 0:
    #     w = - y1 * y2
    #     x = w1 * x2 + y1 * z2
    #     y = w1 * y2
    #     z = w1 * z2 - y1 * x2
    # if x2 == z2 == 0:
    #     w = w1 * w2 - y1 * y2
    #     x = x1 * w2 - z1 * y2
    #     y = w1 * y2 + y1 * w2
    #     z = z1 * w2 + x1 * y2
        
    return w, x, y, z

def q_conjugate(q):
    # https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
    w, x, y, z = q[0], q[1], q[2], q[3]
    return (w, -x, -y, -z)
def qv_mult(q1, v1):
    # https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]



# Let's test my quaternion rotation function

# good_quat = Quaternion([0, 1, 2, 3])
# point = [1.5, 2.5, 3.5]

# my_quat = [0, 1, 2, 3]

# good_rotate = good_quat.rotate(point)
# my_rotate = quat_rotation(normalize(my_quat), point)
# stack_rotate = qv_mult(normalize(my_quat), tuple(point))

