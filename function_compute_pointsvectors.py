#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:10:28 2017

@author: bgris

Computes base points x_i and vectors n_k necessary to launch
estimate_linear_coeff from points a, b, c and from width

"""


import numpy as np
import structured_vector_fields as struct
import generate_data_doigt as gen

def compute_pointsvectors_2articulations(a, b, c, width, sigma):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(b, c)

    nb_ab = int((ab_norm + 0.2*width) / sigma) +1
    nb_ab_orth = int(2 * width / sigma) +1
    nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
    nb_bc_orth = int(2*width / sigma) +1

    points = []

    for i in range(nb_ab + 3):
        for j in range(nb_ab_orth + 3):
            points_ij = [a[u]  + (i-2)*sigma*vector_ab_unit[u] + ((j-2)*sigma - 0.5*width)*vector_ab_norm_orth[u]
                           for u in range(dim)]
            points.append(points_ij.copy())

    for i in range(1, nb_bc + 2):
        for j in range(0, nb_bc_orth + 3):
            points_ij = [b[u] + i*sigma*vector_bc_unit[u] + ((j-2)*sigma - 0.5*width)*vector_bc_norm_orth[u]
                           for u in range(dim)]
            points.append(points_ij.copy())

    vectors = np.array([vector_ab_unit.copy(), vector_ab_norm_orth.copy(),
               vector_bc_unit.copy(), vector_bc_norm_orth.copy()]).T


    return [np.array(points).T, vectors]


def compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(b, c)


    points = []

    for i in range(nb_ab + 3):
        for j in range(nb_ab_orth + 3):
            points_ij = [a[u] + (i-2)*sigma*vector_ab_unit[u] + ((j-2) *sigma- 0.5* width)*vector_ab_norm_orth[u]
                           for u in range(dim)]
            points.append(points_ij.copy())

    for i in range(1, nb_bc + 2):
        for j in range(0, nb_bc_orth + 3):
            points_ij = [b[u] + i*sigma*vector_bc_unit[u] + ((j-2) *sigma- 0.5*width)*vector_bc_norm_orth[u]
                           for u in range(dim)]
            points.append(points_ij.copy())

    vectors = np.array([vector_ab_unit.copy(), vector_ab_norm_orth.copy(),
               vector_bc_unit.copy(), vector_bc_norm_orth.copy()]).T


    return [np.array(points).T, vectors]
