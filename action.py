import numpy as np
import group
import structured_vector_fields

def apply_element_to_field(g, group_element, structured_field):
    """
    structure_field is a matrix (np.array) with 2*dim lines and
    nb_control_points columns
    g is a group
    signed_group_element is a generic signed group
    """
    transformed_field = np.empty_like(structured_field)
    points = structured_vector_fields.get_points(structured_field)
    vectors = structured_vector_fields.get_vectors(structured_field)
    dim, nb_points = vectors.shape


    transformed_points = g.apply(group_element, points)
    transformed_vectors = g.apply_differential(group_element, vectors)

    transformed_field[0:dim] = transformed_points.copy()
    transformed_field[dim:2*dim] = transformed_vectors.copy()

    return transformed_field




