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

def compute_vect_unit(a, b):
    """
    computes unit vectors colin to ab and its orthogonal """
    vector_ab = np.array([bu - au for au, bu in zip(a, b)])
    vector_ab_norm = np.sqrt(sum( [u**2 for u in vector_ab]))
    vector_ab_unit = vector_ab / vector_ab_norm
    vector_ab_norm_orth = np.array([-vector_ab[1], vector_ab[0]])/vector_ab_norm
    return [vector_ab_unit, vector_ab_norm_orth, vector_ab_norm]

def compute_pointsvectors_2articulations(a, b, c, width, sigma):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = compute_vect_unit(b, c)

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
    vector_ab_unit, vector_ab_norm_orth, ab_norm = compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = compute_vect_unit(b, c)


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



def compute_pointsvectors_rectangle(a, b, width, sigma):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = compute_vect_unit(a, b)

    nb_ab = int((ab_norm + 0.4*ab_norm) / sigma) +1
    nb_ab_orth = int(2 * width / sigma) +1
    points = []

    for i in range(nb_ab + 3):
        for j in range(nb_ab_orth + 3):
            points_ij = [a[u]  + ((i-2)*sigma - 0.2*ab_norm)*vector_ab_unit[u] + ((j-2)*sigma - 0.5*width)*vector_ab_norm_orth[u]
                           for u in range(dim)]
            points.append(points_ij.copy())


    vectors = np.array([vector_ab_unit.copy(), vector_ab_norm_orth.copy()]).T


    return [np.array(points).T, vectors]


def compute_pointsvectors_rectangle_nb(a, b, width, sigma, nb_ab, nb_ab_orth):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = compute_vect_unit(a, b)

    points = []

    for i in range(nb_ab + 3):
        for j in range(nb_ab_orth + 3):
            points_ij = [a[u]  + ((i-2)*sigma - 0.2*ab_norm)*vector_ab_unit[u] + ((j-2)*sigma - 0.5*width)*vector_ab_norm_orth[u]
                           for u in range(dim)]
            points.append(points_ij.copy())


    vectors = np.array([vector_ab_unit.copy(), vector_ab_norm_orth.copy()]).T


    return [np.array(points).T, vectors]



def solve_intersection(x1, u1, x2, u2, r):
    # u1 and u2 are supposed to be unitar
    # we search a point y = x1 + t1 u1 = x2 + t2 u2
    # (intersection of the 2 lines)

    # first check if u1 and u2 are colinear
    v1 = np.array([u1[1], -u1[0]])
    ps_orth = sum([v1[i] * u2[i] for i in range(2)])
    ps_orth_x = sum([v1[i] * (x1[i] -x2[i]) for i in range(2)])

    if ps_orth**2 < 1e-10:
        if ps_orth_x**2 < 1e-10:
            # in this case, x1 - x2, u1 and u2 are colinear :
            # take x1 + r * u1
            y = np.array([x1[i] + r * u1[i] for i in range(2)])
        else:
            raise TypeError(' in solve_intersection : no intersection'
                    '')

    else:
        ps_uu =  sum([u1[i] * u2[i] for i in range(2)])
        ps_x1u1 =  sum([x1[i] * u1[i] for i in range(2)])
        ps_x1u2 =  sum([x1[i] * u2[i] for i in range(2)])
        ps_x2u1 =  sum([x2[i] * u1[i] for i in range(2)])
        ps_x2u2 =  sum([x2[i] * u2[i] for i in range(2)])

        t = ps_x1u1*ps_uu - ps_x1u2 - ps_x2u1*ps_uu + ps_x2u2
        t /= (ps_uu**2 -1)

        y =  np.array([x2[i] + t * u2[i] for i in range(2)])

    return y




def compute_pointsvectors_2articulations_bis(a, b, c, width, sigma):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = compute_vect_unit(b, c)

    base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
    base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]

    base_points_b = []
    base_points_b.append(solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
    base_points_b.append(solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())


    vector_points_b_unit, vector_points_b_norm_orth, points_b_norm = compute_vect_unit(base_points_b[0], base_points_b[1])

    n_orth = int((points_b_norm + 0.2*width) / sigma) +1

    nb_ab = int((ab_norm + 0.2*width) / sigma) +1
    nb_bc = int((bc_norm  + 0.2*width) / sigma) +1

    points = []

    for i in range(n_orth+ 3):
        pts_a = [base_points_a[0][u]  + (i-2)*sigma*vector_ab_norm_orth[u]
                           for u in range(dim)]
        pts_b = [base_points_b[0][u]  + (i-2)*sigma*vector_points_b_unit[u]
                           for u in range(dim)]
        pts_c = [base_points_c[0][u]  + (i-2)*sigma*vector_bc_norm_orth[u]
                           for u in range(dim)]

        vector_ptsab_unit, vector_ptsab_norm_orth, ptsab_norm = compute_vect_unit(pts_a, pts_b)
        vector_ptsbc_unit, vector_ptsbc_norm_orth, ptsbc_norm = compute_vect_unit(pts_b, pts_c)

        for i in range(nb_ab + 6):
            points_ij = [pts_a[u]  + (i-6)*(ptsab_norm / nb_ab)*vector_ptsab_unit[u]
                                                       for u in range(dim)]
            points.append(points_ij.copy())

        for i in range(nb_bc + 6):
            points_ij = [pts_b[u]  + i*(ptsbc_norm / nb_bc)*vector_ptsbc_unit[u]
                                                       for u in range(dim)]
            points.append(points_ij.copy())




    vectors = np.array([vector_ab_unit.copy(), vector_ab_norm_orth.copy(),
               vector_bc_unit.copy(), vector_bc_norm_orth.copy()]).T


    return [np.array(points).T, vectors]



def compute_pointsvectors_2articulations_bis_nb(a, b, c, width, sigma, n_orth, nb_ab, nb_bc):
    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = compute_vect_unit(b, c)

    base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
    base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]

    base_points_b = []
    base_points_b.append(solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
    base_points_b.append(solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())


    vector_points_b_unit, vector_points_b_norm_orth, points_b_norm = compute_vect_unit(base_points_b[0], base_points_b[1])

    points = []

    for i in range(n_orth+ 3):
        pts_a = [base_points_a[0][u]  + (i-2)*sigma*vector_ab_norm_orth[u]
                           for u in range(dim)]
        pts_b = [base_points_b[0][u]  + (i-2)*sigma*vector_points_b_unit[u]
                           for u in range(dim)]
        pts_c = [base_points_c[0][u]  + (i-2)*sigma*vector_bc_norm_orth[u]
                           for u in range(dim)]

        vector_ptsab_unit, vector_ptsab_norm_orth, ptsab_norm = compute_vect_unit(pts_a, pts_b)
        vector_ptsbc_unit, vector_ptsbc_norm_orth, ptsbc_norm = compute_vect_unit(pts_b, pts_c)

        for i in range(nb_ab + 6):
            points_ij = [pts_a[u]  + (i-6)*(ptsab_norm / nb_ab)*vector_ptsab_unit[u]
                                                       for u in range(dim)]
            points.append(points_ij.copy())

        for i in range(nb_bc + 6):
            points_ij = [pts_b[u]  + i*(ptsbc_norm / nb_bc)*vector_ptsbc_unit[u]
                                                       for u in range(dim)]
            points.append(points_ij.copy())





    vectors = np.array([vector_ab_unit.copy(), vector_ab_norm_orth.copy(),
               vector_bc_unit.copy(), vector_bc_norm_orth.copy()]).T


    return [np.array(points).T, vectors]



