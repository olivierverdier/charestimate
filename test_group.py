import group
import numpy as np
import numpy.testing as npt
import scipy.linalg as sl
import pytest

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
