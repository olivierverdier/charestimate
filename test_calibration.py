import numpy as np
import odl
import structured_vector_fields as struct

class Translation():
    def exponential(velocity):
        return velocity

def get_action(space):
    proj = projection_periodicity(space)

    def action(translation, f):
        points = struct.get_points(f)
        vectors = struct.get_vectors(f)
        dim, nb_points = points.shape

        points_translated = np.array([[points[u][v] + translation[u] for v in range(nb_points)] for u in range(dim)])
        points_translated_projected = proj(points_translated)

        return struct.create_structured(points_translated_projected, vectors)

    return action





def projection_periodicity(space):
    maxi = space.max_pt[0]
    mini = space.min_pt[0]
    space_extent = maxi - mini

    def proj(point):
        return np.mod(point- space.min_pt, space_extent) + mini

    return proj

def get_kernel(space):
    proj = projection_periodicity(space)

    scale = 0.5*space.cell_sides
    vol_cell = space.cell_volume
    def kernel(point0, point1):

        point0_periodic =proj(point0)
        point1_periodic = proj(point1)

        #value = 0.0
        mask = np.abs(point0_periodic - point1_periodic.T) < scale

        result = np.zeros_like(mask, dtype=float)
        result[mask] = 1 / vol_cell

        return result

    return kernel

space = odl.uniform_discr(
        min_pt =[-1], max_pt=[1], shape=[8],
        dtype='float32', interp='linear')

kernel = get_kernel(space)
def product(f, g):
    return struct.scalar_product_structured(f, g, kernel)

points = space.points()[::2].T
vectors = np.array([[0, 1, 0, 1,]])
original = struct.create_structured(points, vectors)
action = get_action(space)

translated = action(np.array([.2]), original)

noise = odl.phantom.noise.white_noise(space)*0.1
get_unstructured = struct.get_from_structured_to_unstructured(space, kernel)
noisy = get_unstructured(translated) + noise
get_unstructured(original).show()
noisy.show()