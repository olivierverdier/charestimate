import group
import numpy as np
import numpy.testing as npt
import scipy.linalg as sl
import pytest
import numpy.testing as npt

def test_identity():
    g = group.Displacement()
    computed = g.exponential(g.zero_velocity)
    expected = g.identity
    for i in range(2):
        assert pytest.approx(computed[i]) == expected[i]

def test_rotation():
    g = group.Displacement()
    inf = (np.pi, 0, 0)
    element = g.exponential(inf)
    npt.assert_allclose(g.get_translation(element), 0)
    assert pytest.approx(g.get_rotation(element)) == -np.identity(2)

def test_translation():
    g = group.Displacement()
    trans = (0, 1., 0)
    element = g.exponential(trans)
    npt.assert_allclose(g.get_translation(element), np.array([1.,0]))
    assert pytest.approx(g.get_rotation(element)) == np.identity(2)

def test_exponential():
    g = group.Displacement()
    vec = np.random.randn(3)
    theta = vec[0]
    trans = vec[1:]
    velocity = (0., vec)
    velocity_matrix = np.zeros([3,3])
    velocity_matrix[1,0] = theta
    velocity_matrix[0,1] = -theta
    velocity_matrix[:2,-1] = trans
    expected = sl.expm(velocity_matrix)
    computed = g.exponential((theta, trans[0], trans[1]))
    assert pytest.approx(computed) == expected


def test_sinc():
    assert pytest.approx(group.sinc(np.pi)) == 0

def test_apply():
    points = np.array([[0, 1], [1, 2]])
    g = group.Displacement()
    inf = (0.5*np.pi, 0, 0)
    trans = (0, 1., 0)
    group_element0 = g.exponential(g.zero_velocity)
    group_element1 = g.exponential(inf)
    group_element2 = g.exponential(trans)

    transformed0 = g.apply(group_element0, points)
    transformed1 = g.apply(group_element1, points)
    transformed2 = g.apply(group_element2, points)

    npt.assert_allclose(transformed0, points)

    tranformed1_expected = np.array([[-1, -2], [0, 1]])
    print('expected = {} , computed = {}'.format(transformed1, tranformed1_expected))
    #npt.assert_allclose(transformed1, tranformed1_expected)

    tranformed2_expected = np.array([[1, 2], [1, 2]])
    print('expected = {} , computed = {}'.format(transformed2, tranformed2_expected))
    #npt.assert_allclose(transformed2, tranformed2_expected)

    transformed_diff0 = g.apply_differential(group_element0, points)
    transformed_diff1 = g.apply_differential(group_element1, points)
    transformed_diff2 = g.apply_differential(group_element2, points)

    transformed_diff_transpose0 = g.apply_differential_transpose(group_element0, points)
    transformed_diff_transpose1 = g.apply_differential_transpose(group_element1, points)
    transformed_diff_transpose2 = g.apply_differential_transpose(group_element2, points)


