import numpy as np
import group
sd = group.ScaleDisplacement
import action
import numpy.testing as npt

import accessors as acc

import structured_vector_fields

import odl
import structured_vector_fields as struct

def apply_element_to_field_old(signed_group_element, structured_field):
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

    group_element = acc.get_group_element(signed_group_element)
    epsilon = acc.get_sign(signed_group_element)
    lam = acc.get_scale(group_element)

    displacement = acc.get_rigid(group_element)
    transformed_points = np.dot(displacement, points)
    transformed_vectors = np.dot(group.Displacement.get_rotation(displacement), vectors)

    transformed_field[0:dim] = transformed_points[0:dim].copy()
    transformed_field[dim:2*dim] = lam * epsilon * transformed_vectors.copy()

    return transformed_field

def test_apply_element_to_field():
    scale = 0
    tx = 0.0
    ty = 0.0
    angle=np.pi/2
    infinitesimal = (scale, (angle, tx, ty))
    infinitesimal1 = (1.0, (angle, tx, ty))
    signed_group_element0=(1, sd.exponential(infinitesimal))
    signed_group_element1=(-1, sd.exponential(infinitesimal))
    signed_group_element2=(1, sd.exponential(infinitesimal1))
    structured_field0 = np.array([[0,0], [0,0], [0,1], [1,0]])

    #expected
    structured_field1 = np.array([[0,0], [0,0], [-1,0], [0,1]])
    structured_field2 = np.array([[0,0], [0,0], [1,0], [0,-1]])
    structured_field3 = np.array([[0,0], [0,0], [-np.exp(1),0], [0,np.exp(1)]])

    # obtained
    structured_field4 = apply_element_to_field_old(signed_group_element0, structured_field0)
    structured_field5 = apply_element_to_field_old(signed_group_element1, structured_field0)
    structured_field6 = apply_element_to_field_old(signed_group_element2, structured_field0)

    assert structured_field4.any() == structured_field1.any()

    assert structured_field2.any() == structured_field5.any()

    assert structured_field3.any() == structured_field6.any()

def get_action(space):
    """
    Direct action of translation implemented separately.
    """

    proj = group.projection_periodicity(space)

    def action(translation, f):
        points = struct.get_points(f)
        vectors = struct.get_vectors(f)
        dim, nb_points = points.shape

        points_translated = np.array([[points[u][v] + translation[u] for v in range(nb_points)] for u in range(dim)])
        points_translated_projected = proj(points_translated)

        return struct.create_structured(points_translated_projected, vectors)

    return action

def test_translation_action():
    space = odl.uniform_discr(
            min_pt =[-1], max_pt=[1], shape=[128],
            dtype='float32', interp='linear')

    points = np.array([[-0.75, 0.0, 0.2, 0.5,]])
    vectors = np.array([[0.3, 0.0, 0, 1,]])
    original = struct.create_structured(points, vectors)
    translation_action = get_action(space)

    translation = np.array([1.0])
    expected = translation_action(translation, original)

    g = group.Translation(space)

    computed = action.apply_element_to_field(g, translation, original)

    npt.assert_allclose(computed, expected)
