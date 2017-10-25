#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:30:01 2017

@author: bgris
"""

import numpy as np
import group
import structured_vector_fields as struct
import accessors as acc


def solve_regression(g, group_element_list, field_list, sigma0, sigma1, points, eval_kernel):
    # We suppose here that the kernel is invariant wrt the group action
    nb_data=len(group_element_list)
    dim, nb_points=points.shape

    vector_syst = np.zeros(dim*nb_points)
    basis = np.identity(dim)

    # computes \sum_i A_i^T A_i e_l for each l
    basis_deformed_tot = np.zeros_like(basis)

    for i in range(nb_data):
        field_i = field_list[i].copy()
        group_element_i =  group_element_list[i]
        points_transf_i = g.apply(group_element_i, points)
        basis_transf_i =  g.apply_differential(group_element_i, basis)
        basis_deformed_tot += g.apply_differential_transpose(group_element_i, basis_transf_i)

        eval_field_i = np.array([field_i[u].interpolation(
                points_transf_i) for u in range(dim)])


        # TODO: use broadcasting here
        for k0 in range(nb_points):
            for l0 in range(dim):
                vector_syst[dim*k0 + l0] += np.dot(eval_field_i.T[k0],
                        basis_transf_i[:, l0]) / (sigma0 ** 2)

    matrix_syst = np.kron(eval_kernel,
        (nb_points/(sigma1 ** 2)) * basis + (1/(sigma0 ** 2)) * basis_deformed_tot)


    return np.linalg.solve(matrix_syst, vector_syst)














