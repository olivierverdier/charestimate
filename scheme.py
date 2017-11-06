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

def calibrate_list(original, field_list, calibration):
    nb_data = len(field_list)
    group_element_list = [calibration(original, field_list[i]) for i in range(nb_data)]
    return group_element_list


def iterative_scheme(solve_regression, calibration, action, g, kernel, field_list, sigma0, sigma1, points, nb_iteration):
    nb_data = len(field_list)
    eval_kernel = struct.make_covariance_matrix(points, kernel)
    dim, nb_points = points.shape
    def product(vect0, vect1):
        return struct.scalar_product_structured(vect0, vect1, kernel)

    # initialization with a structured version of first vector field (NOT GOOD)
    group_element_init = g.identity
    vectors_original = solve_regression(g, [group_element_init], [field_list[0]], sigma0, sigma1, points, eval_kernel)
    vectors_original_struct = struct.get_structured_vectors_from_concatenated(vectors_original, nb_points, dim)
    original = struct.create_structured(points, vectors_original_struct)
    get_unstructured_op = struct.get_from_structured_to_unstructured(field_list[0].space[0], kernel)
    get_unstructured_op(original).show('initialisation')

    for k in range(nb_iteration):
        velocity_list = calibrate_list(original, field_list, calibration)
        group_element_list = [g.exponential(velocity_list[i]) for i in range(nb_data)]
        vectors_original = solve_regression(g, group_element_list, field_list, sigma0, sigma1, points, eval_kernel)
        vectors_original_struct = struct.get_structured_vectors_from_concatenated(vectors_original, nb_points, dim)
        original = struct.create_structured(points, vectors_original_struct)
        print('iteration {}'.format(k))
        get_unstructured_op(original).show('iteration {}'.format(k))

    return [original, group_element_list]


#
#
#    def action_fun(signed_group_element, structured_field):
#        return action.apply_element_to_field(g, signed_group_element, structured_field)
