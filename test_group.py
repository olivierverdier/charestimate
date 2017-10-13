import group
import numpy as np
import numpy.testing as npt
import scipy.linalg as sl
import pytest

def test_identity():
    computed = group.exponential(group.zero_velocity)
    expected = group.identity
    for i in range(2):
        assert pytest.approx(computed[i]) == expected[i]

def test_rotation():
    inf = (0, (np.pi, 0, 0))
    element = group.exponential(inf)
    npt.assert_allclose(group.get_rigid(element)[:-1,-1], 0)
    assert pytest.approx(group.get_rotation(element)) == -np.identity(2)

def test_translation():
    trans = (0, (0, 1., 0))
    element = group.exponential(trans)
    npt.assert_allclose(group.get_rigid(element)[:-1,-1], np.array([1.,0]))
    assert pytest.approx(group.get_rotation(element)) == np.identity(2)

def test_exponential():
    vec = np.random.randn(3)
    theta = vec[0]
    trans = vec[1:]
    velocity = (0., vec)
    velocity_matrix = np.zeros([3,3])
    velocity_matrix[1,0] = theta
    velocity_matrix[0,1] = -theta
    velocity_matrix[:2,-1] = trans
    expected = sl.expm(velocity_matrix)
    computed = group.get_rigid(group.exponential((0, (theta, trans[0], trans[1]))))
    assert pytest.approx(computed) == expected


def test_sinc():
    assert pytest.approx(group.sinc(np.pi)) == 0
