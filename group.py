import numpy as np

class Group():
    pass

class ContinuousGroup(Group):
    pass

class Displacement(ContinuousGroup):
    zero_velocity = (0, 0, 0)
    identity = np.identity(3)

    @classmethod
    def exponential(self, velocity):
        angle, tx, ty = velocity
        trans = np.array([tx, ty])
        rot = rotation_matrix(angle)
        rot2 = rotation_matrix(angle/2)
        coeff = sinc(angle/2)
        rotated_trans = np.dot(rot2, trans)
        matrix = np.zeros([3,3])
        matrix[:2,:2] = rot
        matrix[-1,-1] = 1.
        matrix[:-1, -1] = coeff * rotated_trans
        return matrix

    @classmethod
    def get_translation(self, matrix):
        return matrix[:-1,-1]

    @classmethod
    def get_rotation(self, group_element):
        return group_element[:2,:2]

class Scaling(ContinuousGroup):
    zero_velocity = 0
    identity = 1.

    @classmethod
    def exponential(self, velocity):
        return np.exp(velocity)

def make_product(G1, G2):
    class Product(ContinuousGroup):
        zero_velocity = (G1.zero_velocity, G2.zero_velocity)
        identity = (G1.identity, G2.identity)

        @classmethod
        def exponential(self, velocity):
            v1, v2 = velocity
            return (G1.exponential(v1), G2.exponential(v2))
    return Product



ScaleDisplacement = make_product(Scaling, Displacement)


def sinc(x):
    return np.sinc(x/np.pi)

def rotation_matrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, s],[-s,c]]).T


