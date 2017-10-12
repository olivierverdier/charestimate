import numpy as np

zero_velocity = (0, (0, 0, 0))

identity = (1., np.identity(3))

def sinc(x):
    return np.sinc(x/np.pi)

def rotation_matrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, s],[-s,c]]).T

def exponential(infinitesimal):
    scale, (angle, tx, ty) = infinitesimal
    trans = np.array([tx, ty])
    rot = rotation_matrix(angle)
    rot2 = rotation_matrix(angle/2)
    coeff = sinc(angle/2)
    rotated_trans = np.dot(rot2, trans)
    matrix = np.zeros([3,3])
    matrix[:2,:2] = rot
    matrix[-1,-1] = 1.
    matrix[:-1, -1] = coeff * rotated_trans
    return (np.exp(scale), matrix)

def get_rotation(group_element):
    return group_element[:2,:2]
