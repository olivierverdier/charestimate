#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:13:22 2017

@author: bgris

The function takes in input a kernel K and time series of
vector fields v_l, of base points x_li and of vectors n_lk;
and returns scalars alpha_ik (independent from time) such that the quantity
\sum_l || v_l - \sum_{i,k} K(x_li, .) alpha_ik n_lk ||^2_V is minimal
(with V the RKHS of kernel K)

"""

import numpy as np
import structured_vector_fields as struct

def estimate_linear_coeff(kernel, vector_fields_list, points_list, vectors_list):
    """
    vector_fields_list : list of space.tangent_bundle.element
    points_list : list of arrays of size dim x Nb_points
    vectors_list : list of arrays of size dim x Nb_vectors


    """

    Nb_time = len(vector_fields_list)
    Nb_points = points_list[0].shape[1]
    Nb_vectors = vectors_list[0].shape[1]

    cov_matrix = np.zeros([Nb_points * Nb_vectors, Nb_points * Nb_vectors])
    beta = np.zeros(Nb_points * Nb_vectors)
    for l in range(Nb_time):
        vector_fields_l = vector_fields_list[l].copy()
        points_l = points_list[l].copy()
        vectors_l = vectors_list[l].copy()
        cov_mat_l = struct.make_covariance_matrix(points_l, kernel)
        mat_prod_vectors = np.dot(vectors_l.T, vectors_l)
        cov_matrix += np.kron(cov_mat_l, mat_prod_vectors)
        vector_field_interp = np.array(
                [v.interpolation(points_l) for v in vector_fields_l])

        beta += sum(np.kron(v, vectors) for v, vectors in zip(vector_field_interp, vectors_l))

    return np.linalg.solve(cov_matrix, beta)


