import numpy as np
import accessors as acc

class Group():
    pass

class ContinuousGroup(Group):
    pass
# TODO: make a difference between continuous and discrete groups





def projection_periodicity(space):
    """
    Return: periodic projection of a point on the space.
    """
    maxi = space.max_pt[0]
    mini = space.min_pt[0]
    space_extent = maxi - mini
    def proj(point):
        return np.mod(point- space.min_pt, space_extent) + mini
    return proj

class Translation():
    def __init__(self, space):
        self.space = space

    def exponential(self, velocity):
        return velocity
    zero_velocity = np.array([0.0])
    identity = np.array([0.0])

    def apply(self, translation, points):
        dim, nb_points = points.shape
        proj = projection_periodicity(self.space)
        points_translated = np.array([[points[u][v] + translation[u] for v in range(nb_points)] for u in range(dim)])
        points_translated_projected = proj(points_translated)
        return points_translated_projected

    def apply_differential(self, group_element, vectors):
        return vectors

    def apply_differential_transpose(self, group_element, vectors):
        return vectors

class Displacement(ContinuousGroup):
    zero_velocity = np.array([0, 0, 0])
    identity = np.identity(3)

    @classmethod
    def exponential(self, velocity):
        angle, tx, ty = velocity
        trans = np.array([tx, ty])
        rot = rotation_matrix(angle)
        rot2 = rotation_matrix(angle/2)
        coeff = sinc(angle/2)
        rotated_trans = np.dot(rot2, trans)
        matrix = np.zeros([3, 3])
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

    @classmethod
    def apply(self, group_element, points):
        dim, nb_points = points.shape
        points_homogeneous = np.empty([dim + 1, nb_points])
        points_homogeneous[0:dim,:] = points.copy()
        points_homogeneous[dim,:] = 1.0

        return np.dot(group_element, points_homogeneous)[0:2, :]

    @classmethod
    def apply_differential(self, group_element, vectors):
        dim, nb_points = vectors.shape
        rotation = self.get_rotation(group_element)

        return np.dot(rotation, vectors)


    @classmethod
    def apply_differential_transpose(self, group_element, vectors):
        dim, nb_points = vectors.shape
        rotation = self.get_rotation(group_element)

        return np.dot(rotation.T, vectors)

    @classmethod
    def to_array(self, velocity):
        return np.array(velocity)


class Scaling(ContinuousGroup):
    zero_velocity = np.array([0.])
    identity = np.array([1.])

    @classmethod
    def exponential(self, velocity):
        return np.exp(velocity)

    @classmethod
    def apply(self, group_element, points):
        return points.copy()

    @classmethod
    def apply_differential(self, group_element, vectors):
        return group_element * vectors.copy()

    @classmethod
    def apply_differential_transpose(self, group_element, vectors):
        return group_element * vectors.copy()


def make_product(G1, G2):
    class Product(ContinuousGroup):
        zero_velocity_1 = G1.zero_velocity
        dim1 = len(zero_velocity_1)
        zero_velocity_2 = G2.zero_velocity
        dim2 = len(zero_velocity_2)
        zero_velocity = np.array(list(zero_velocity_1) + list(zero_velocity_2))
        identity = (G1.identity, G2.identity)

        @classmethod
        def exponential(self, velocity):
            v1 = velocity[:self.dim1]
            v2 = velocity[self.dim1:]
            return (G1.exponential(v1), G2.exponential(v2))

        @classmethod
        def apply(self, group_element, points):
            g1, g2 = group_element
            return G1.apply(g1, G2.apply(g2, points))

        @classmethod
        def apply_differential(self, group_element, vectors):
            g1, g2 = group_element
            return G1.apply_differential(g1, G2.apply_differential(g2,
                                                                   vectors))

        @classmethod
        def apply_differential_transpose(self, group_element, vectors):
            g1, g2 = group_element
            return G1.apply_differential_transpose(g1,
                               G2.apply_differential_transpose(g2, vectors))

    return Product


ScaleDisplacement = make_product(Scaling, Displacement)
ScaleTranslation = make_product(Scaling, Translation)


def sinc(x):
    return np.sinc(x/np.pi)


def rotation_matrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, s], [-s, c]]).T
