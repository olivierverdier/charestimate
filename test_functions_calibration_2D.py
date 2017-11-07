import numpy as np
import odl
import structured_vector_fields as struct
import calibration as calib
import group
import action
import functions_calibration_2D as func_2D


def kernel(x,y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x,y)]) / (sigma ** 2))

space = odl.uniform_discr(
        min_pt =[-10,-10], max_pt=[10,10], shape=[128, 128],
        dtype='float32', interp='linear')

function = func_2D.function_2D_scalingdisplacement(space, kernel)

def test_function_2D_scalingdisplacement_solver():
    g = group.ScaleDisplacement()

    vec = np.random.randn(4)
    group_el = g.exponential(vec)

    w1 = np.random.randn(2, 2)

    w2 = [group_el[0] * np.dot(group_el[1][0:2, 0:2], w1[0]),
          np.dot(group_el[1][0:2, 0:2], w1[1]) + group_el[1][0:2, 2]]

    computed = function.solver(w1, w2)

    print('real = {}'.format(vec))
    print('computed = {}'.format(computed))