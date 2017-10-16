#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:30:01 2017

@author: bgris
"""

import odl
import numpy as np
import group
import structured_vector_fields as struct


def solve_regression(signed_group_element_list, field_list, sigma0, sigma1, points, eval_kernel):
    # We suppose here that the kernel is invariant wrt the group action
    nb_data=len(signed_group_element_list)
    dim, nb_points=points.shape


    points_homogeneous = np.empty([dim + 1, nb_points])
    points_homogeneous[0:dim,:] = points.copy()
    points_homogeneous[dim,:] = 1.0
    vector_syst = np.zeros(dim*nb_points)
    basis = np.identity(dim)

    sum_lam = 0

    for i in range(nb_data):
        signed_group_element_i =  signed_group_element_list[i]
        field_i = field_list[i].copy()
        epsilon_i = group.get_sign(signed_group_element_i)
        group_element_i = group.get_group_element(signed_group_element_i)
        rotation_i = group.get_rotation(group_element_i)
        lam_i = group.get_scale(group_element_i)
        rigid_i = group.get_rigid(group_element_i)
        points_transf_i = np.dot(rigid_i, points_homogeneous)[0:2, :]
        eval_field_i = np.array([field_i[u].interpolation(
                points_transf_i) for u in range(dim)])

        sum_lam += lam_i * lam_i

        for k0 in range(nb_points):
            for l0 in range(dim):
                vector_syst[dim*k0 + l0]+= np.dot(eval_field_i[:,k0], 
                        epsilon_i * lam_i * rotation_i[:,l0]) / (sigma0 ** 2)

    matrix_syst = ((sum_lam/(sigma0 ** 2)) + (nb_points/(sigma1 ** 2))) *  np.kron(eval_kernel, basis)


    return np.linalg.solve(matrix_syst, vector_syst)






