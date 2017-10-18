import numpy as np
import group
import structured_vector_fields
from accessors import *

def apply_element_to_field(signed_group_element, structured_field):
    """
    structure_field is a matrix (np.array) with 2*dim lines and
    nb_control_points columns
    signed_group_element is [epsilon, exp] with exp=[lambda,
    matrix_rigid_defo]
    """
    transformed_field = np.empty_like(structured_field)
    points = structured_vector_fields.get_homogeneous_points(structured_field)
    vectors = structured_vector_fields.get_vectors(structured_field)
    dim, nb_points = vectors.shape

    group_element = get_group_element(signed_group_element)
    epsilon = get_sign(signed_group_element)
    lam = get_scale(group_element)

    displacement = get_rigid(group_element)
    transformed_points = np.dot(displacement, points)
    transformed_vectors = np.dot(group.Displacement.get_rotation(displacement), vectors)

    transformed_field[0:dim] = transformed_points[0:dim].copy()
    transformed_field[dim:2*dim] = lam * epsilon * transformed_vectors.copy()

    return transformed_field
