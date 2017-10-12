import pytest
from pytest import approx
import group

def test_identity():
    assert approx(group.exponential(group.zero_velocity)) == group.identity

def test_sinc():
    assert pytest.approx(group.sinc(np.pi)) == 0
