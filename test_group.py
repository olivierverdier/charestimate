import pytest
from pytest import approx
import group

def test_identity():
    assert approx(group.exponential(group.zero_velocity)) == group.identity
