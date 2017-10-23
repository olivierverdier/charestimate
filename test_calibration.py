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

        points_translated = points + translation
        points_translated_projected = proj(points_translated)

        return struct.create_structured(points_translated_projected, vectors)

    return action





def projection_periodicity(space):

    space_extent = space.max_pt - space.min_pt

    def proj(point):
        return np.mod(point, space_extent) + space.min_pt

    return proj

def get_kernel(space):
    proj = projection_periodicity(space)

    scale = space.cell_sides
    vol_cell = space.cell_volume
    def kernel(point0, point1):

        point0_periodic =proj(point0)
        point1_periodic = proj(point1)

        value = 0.0
        if (np.abs(point0_periodic - point1_periodic) < scale):
            value = (1 / vol_cell)

        return value

    return kernel

space = odl.uniform_discr(
        min_pt =[-1], max_pt=[1], shape=[8],
        dtype='float32', interp='linear')

kernel = get_kernel(space)
def product(f, g):
    return struct.scalar_product_structured(f, g, kernel)

points = space.points()[::2]
vectors = np.ones_like(points)
original = struct.create_structured(points, vectors)
action = get_action(space)

translated = action(.5, original)
vs = struct.get_vectors()

