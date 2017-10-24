#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:36:24 2017

@author: bgris
"""

import numpy as np
import structured_vector_fields as struct
import calibration as calib
import accessors as acc

def calibrate_list(original, field_list, g, action, product, pairing):
    nb_data = len(field_list)
    group_element_list = [calib.calibrate(original, field_list[i], g, action, product, pairing).x for i in range(nb_data)]
    return [acc.create_signed_element(1, group_element_list[i]) for i in range(nb_data)]


def iterative_scheme(solve_regression, calibration, action, g, kernel, field_list, sigma0, sigma1, points, nb_iteration):

    eval_kernel = struct.make_covariance_matrix(points, kernel)

    def product(vect0, vect1):
        return struct.scalar_product_structured(vect0, vect1, kernel)

    # initialization with a structured version of first vector field (NOT GOOD)
    signed_group_element_init = acc.create_signed_element(1,  g.identity)
    vectors_original = solve_regression(g, [signed_group_element_init], [field_list[0]], sigma0, sigma1, points, eval_kernel)
    original = struct.create_structured(points, vectors_original)

    for k in range(nb_iteration):
        signed_group_element_list = calibrate_list(original, field_list, g, action, product, struct.scalar_product_unstructured)
        vectors_original = solve_regression(g, signed_group_element_list, field_list, sigma0, sigma1, points, eval_kernel)
        original = struct.create_structured(points, vectors_original)

    return [original, signed_group_element_list]


#
#
#    def action_fun(signed_group_element, structured_field):
#        return action.apply_element_to_field(g, signed_group_element, structured_field)
