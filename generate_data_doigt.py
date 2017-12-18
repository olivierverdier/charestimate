#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:24:15 2017

@author: bgris
"""

import structured_vector_fields
import group
import pytest
import numpy as np
import odl
import estimate_structured_base_pointsvectors as est_coeff

#%% functions
def Rtheta(theta,points):
    # theta is the angle, in rad
    # input = list of points, for ex given by space.points() or
    # np.array(vect_field).T
    #output = list of points of same size, rotated of an angle theta

    points_rot=np.empty_like(points).T
    points_rot[0]=np.cos(theta)*points.T[0].copy() - np.sin(theta)*points.T[1].copy()
    points_rot[1]=np.sin(theta)*points.T[0].copy() + np.cos(theta)*points.T[1].copy()

    return points_rot.T.copy()
#


def Rot_inf(points):
    # theta is the angle, in rad
    # input = list of points, for ex given by space.points() or
    # np.array(vect_field).T
    #output = list of points of same size, rotated of an angle theta

    points_rot=np.empty_like(points).T
    points_rot[0] = -points.T[1].copy()
    points_rot[1] = points.T[0].copy()

    return points_rot.T.copy()
#

def compute_vect_unit(a, b):
    """
    computes unit vectors colin to ab and its orthogonal """
    vector_ab = np.array([bu - au for au, bu in zip(a, b)])
    vector_ab_norm = np.sqrt(sum( [u**2 for u in vector_ab]))
    vector_ab_unit = vector_ab / vector_ab_norm
    vector_ab_norm_orth = np.array([-vector_ab[1], vector_ab[0]])/vector_ab_norm
    return [vector_ab_unit, vector_ab_norm_orth, vector_ab_norm]

def generate_image_2articulations(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim=2
    points=space.points().T
    limit = 0.0*width

    vector_ab_unit, vector_ab_norm_orth, vector_ab_norm = compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, vector_bc_norm = compute_vect_unit(b, c)

    points_prod_ab = sum([(points[u] - a[u])*vector_ab_unit[u] for u in range(dim)])
    points_prod_bc = sum([(points[u] - b[u])*vector_bc_unit[u]  for u in range(dim)])

    points_prod_ab_orth = sum([(points[u] - a[u])*vector_ab_norm_orth[u] for u in range(dim)])
    points_prod_bc_orth = sum([(points[u] - b[u])*vector_bc_norm_orth[u]  for u in range(dim)])

    I_arti0 = (0-limit <= points_prod_ab )*(points_prod_ab <= vector_ab_norm + limit)
    I_arti0 *= (points_prod_ab_orth >= 0 - limit)* (points_prod_ab_orth <= width + limit)

    I_arti1 = (0 - limit<= points_prod_bc )*(points_prod_bc <= vector_bc_norm + limit)
    I_arti1 *= (points_prod_bc_orth >= 0-limit)* (points_prod_bc_orth <= width + limit)

    return space.element((I_arti0 == 1) + (I_arti1 == 1))
#

def generate_image_2articulations_vectfield(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim=2
    points=space.points().T
    limit = 0.3*width

    vector_ab_unit, vector_ab_norm_orth, vector_ab_norm = compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, vector_bc_norm = compute_vect_unit(b, c)

    points_prod_ab = sum([(points[u] - a[u])*vector_ab_unit[u] for u in range(dim)])
    points_prod_bc = sum([(points[u] - b[u])*vector_bc_unit[u]  for u in range(dim)])

    points_prod_ab_orth = sum([(points[u] - a[u])*vector_ab_norm_orth[u] for u in range(dim)])
    points_prod_bc_orth = sum([(points[u] - b[u])*vector_bc_norm_orth[u]  for u in range(dim)])

    I_arti0 = (0-limit <= points_prod_ab )*(points_prod_ab <= vector_ab_norm + limit)
    I_arti0 *= (points_prod_ab_orth >= 0 - limit)* (points_prod_ab_orth <= width + limit)

    I_arti1 = (0 - limit<= points_prod_bc )*(points_prod_bc <= vector_bc_norm + limit)
    I_arti1 *= (points_prod_bc_orth >= 0-limit)* (points_prod_bc_orth <= width + limit)

    return space.element((I_arti0 == 1) + (I_arti1 == 1))
#


def generate_vectorfield_2articulations_0(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim = 2
    points=space.points().T
    I = generate_image_2articulations_vectfield(space, a, b, c, width)
    points_a = np.array([points[u] - a[u] for u in range(dim)])
    vect = space.tangent_bundle.element(Rot_inf(points_a.T).T)

    return vect*I

#

#%% generate data


space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[128, 128],
        dtype='float32', interp='linear')


width = 1
vector_fields_list = []
image_list = []
a_list = [[0,0], [0,0], [0,0]]
b_list = [[0, 2], [0, 2], [-2, 0]]
c_list = [[0, 5], [-5, 2], [-5, 0]]
nbdata = 3

for i in range(nbdata):
    a = a_list[i]
    b = b_list[i]
    c = c_list[i]
    vector_fields_list.append(generate_vectorfield_2articulations_0(space, a, b, c, width))
    image_list.append(generate_image_2articulations(space, a, b, c, width))
#
