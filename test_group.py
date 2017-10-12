import group
import numpy as np
import numpy.testing as npt
import pytest

def test_identity():
    computed = group.exponential(group.zero_velocity)
    expected = group.identity
    for i in range(2):
        assert pytest.approx(computed[i]) == expected[i]

def test_rotation():
    inf = (0, (np.pi, 0, 0))
    element = group.exponential(inf)[1]
    npt.assert_allclose(element[:-1,-1], 0)
    assert pytest.approx(group.get_rotation(element)) == -np.identity(2)

def test_translation():
    trans = (0, (0, 1., 0))
    element = group.exponential(trans)[1]
    npt.assert_allclose(element[:-1,-1], np.array([1.,0]))
    assert pytest.approx(group.get_rotation(element)) == np.identity(2)

def test_sinc():
    assert pytest.approx(group.sinc(np.pi)) == 0
