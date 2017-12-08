#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:40:15 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:48:04 2017

@author: bgris
"""

import structured_vector_fields
import group
import pytest
import numpy as np
import odl
import numpy.testing as npt
import estimate_structured_base_pointsvectors as est_coeff




space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[128, 128],
        dtype='float32', interp='linear')

sigma_kernel = 2
def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x,y)]) / (sigma_kernel ** 2))


def estimate_linear_coeff():

    vector_fields_list =[space.tangent_bundle.one(), 2*space.tangent_bundle.one()]
    points_list = [np.array([[0], [0]]), np.array([[0], [0]])]

    vectors_list0 = [np.array([[1], [1]]), np.array([[2], [2]])]
    coeff_exp0 = [1]
    coeff_computed0 = est_coeff.estimate_linear_coeff(kernel, vector_fields_list, points_list, vectors_list0)
    print('expected0 {}'.format(coeff_exp0))
    print('computed0 {}'.format(coeff_computed0))

    vectors_list1 = [np.array([[1, 0], [0, 2]]), np.array([[2, 0], [0, 4]])]
    coeff_exp1 = [1, 0.5]
    coeff_computed1 = est_coeff.estimate_linear_coeff(kernel, vector_fields_list, points_list, vectors_list1)
    print('expected1 {}'.format(coeff_exp1))
    print('computed1 {}'.format(coeff_computed1))

    points_list2 =[np.array([[-8, 8], [-8, 8]]), np.array([[-8, 8], [-8, 8]])]
    vectors_list2 = [np.array([[1], [1]]), np.array([[2], [2]])]
    coeff_exp2 = [1, 1]
    coeff_computed2 = est_coeff.estimate_linear_coeff(kernel, vector_fields_list, points_list2, vectors_list2)
    print('expected2 {}'.format(coeff_exp2))
    print('computed2 {}'.format(coeff_computed2))

estimate_linear_coeff()

#