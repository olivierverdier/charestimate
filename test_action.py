import numpy as np
import group
sd = group.ScaleDisplacement
import action

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
    structured_field4 = action.apply_element_to_field(signed_group_element0, structured_field0)
    structured_field5 = action.apply_element_to_field(signed_group_element1, structured_field0)
    structured_field6 = action.apply_element_to_field(signed_group_element2, structured_field0)

    assert structured_field4.any() == structured_field1.any()

    assert structured_field2.any() == structured_field5.any()

    assert structured_field3.any() == structured_field6.any()
