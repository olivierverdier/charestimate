import numpy as np
import group
import structured_vector_fields
from accessors import *

def apply_element_to_field(g, signed_group_element, structured_field):
    """
    structure_field is a matrix (np.array) with 2*dim lines and
    nb_control_points columns
    g is a group
    signed_group_element is a generic signed group
    """
    transformed_field = np.empty_like(structured_field)
    points = structured_vector_fields.get_homogeneous_points(structured_field)
    vectors = structured_vector_fields.get_vectors(structured_field)
    dim, nb_points = vectors.shape

    group_element = get_group_element(signed_group_element)
    epsilon = get_sign(signed_group_element)

    transformed_points = g.apply(group_element, points)
    transformed_vectors = epsilon * g.apply_differential(group_element, vectors)

    transformed_field[0:dim] = transformed_points.copy()
    transformed_field[dim:2*dim] = transformed_vectors.copy()

    return transformed_field




