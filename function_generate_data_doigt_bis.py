#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:12:10 2018

@author: bgris
"""


import numpy as np
import structured_vector_fields as struct

import function_compute_pointsvectors as cmp

##%% functions
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

def generate_image_2articulations(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim=2
    points=space.points().T
    #limit = 0.0*width

    vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(b, c)
    base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
    base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]

    base_points_b = []
    base_points_b.append(cmp.solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
    base_points_b.append(cmp.solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())


    limit_ab  = 0.2*ab_norm
    limit_bc  = 0.2*bc_norm
    limit_orth = 0.2*width



    points_prod_0_ab = sum([(points[u] - base_points_b[0][u])*vector_ab_unit[u] for u in range(dim)])
    points_prod_0_bc = sum([(points[u] - base_points_b[0][u])*vector_bc_unit[u]  for u in range(dim)])

    points_prod_0_ab_orth = sum([(points[u] - base_points_b[0][u])*vector_ab_norm_orth[u] for u in range(dim)])
    points_prod_0_bc_orth = sum([(points[u] - base_points_b[0][u])*vector_bc_norm_orth[u]  for u in range(dim)])


    I_arti0 = (-ab_norm - limit_ab <= points_prod_0_ab )*(points_prod_0_ab <= 0 + 0*limit_ab)
    I_arti0 *= (0 <= points_prod_0_ab_orth )*(points_prod_0_ab_orth <= width + 0*limit_orth)


    I_arti1 = (0 <= points_prod_0_bc )*(points_prod_0_bc <= bc_norm + limit_bc)
    I_arti1 *= (0 <= points_prod_0_bc_orth )*(points_prod_0_bc_orth <= width + 0*limit_orth)

    return space.element((I_arti0 == 1) + (I_arti1 == 1))
#

def generate_image_2articulations_vectfield(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    used to generate vector field in generate_vectorfield_2articulations_0
    """
    dim=2
    points=space.points().T
    limit = 0.3*width

    vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(b, c)
    base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
    base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]

    base_points_b = []
    base_points_b.append(cmp.solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
    base_points_b.append(cmp.solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())


    limit_ab  = 0.2*ab_norm
    limit_bc  = 0.2*bc_norm
    limit_orth = 0.2*width



    points_prod_0_ab = sum([(points[u] - base_points_b[0][u])*vector_ab_unit[u] for u in range(dim)])
    points_prod_0_bc = sum([(points[u] - base_points_b[0][u])*vector_bc_unit[u]  for u in range(dim)])

    points_prod_0_ab_orth = sum([(points[u] - base_points_b[0][u])*vector_ab_norm_orth[u] for u in range(dim)])
    points_prod_0_bc_orth = sum([(points[u] - base_points_b[0][u])*vector_bc_norm_orth[u]  for u in range(dim)])


    I_arti0 = (-ab_norm - limit_ab - limit<= points_prod_0_ab )*(points_prod_0_ab <= 0 + 0*limit_ab + limit)
    I_arti0 *= (0 - limit <= points_prod_0_ab_orth )*(points_prod_0_ab_orth <= width + 0*limit_orth + limit)


    I_arti1 = (0 - limit<= points_prod_0_bc )*(points_prod_0_bc <= bc_norm + limit_bc + limit)
    I_arti1 *= (0 - limit <= points_prod_0_bc_orth )*(points_prod_0_bc_orth <= width + 0*limit_orth + limit)

    return space.element((I_arti0 == 1) + (I_arti1 == 1))
#
#


def generate_vectorfield_2articulations_0(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a vector field corresponding to rotation on the 'finger
    ' with 2 articulations at a and b, with ending point at c,
    rotating the whole finger (1st 'articulation)
    """
    vector_ab_unit, vector_ab_norm_orth, vector_ab_norm = cmp.compute_vect_unit(a, b)
    limit_ab = 0.2*vector_ab_norm
    centre_rot = a - limit_ab*vector_ab_norm_orth
    dim = 2
    points = space.points().T
    I = generate_image_2articulations_vectfield(space, a, b, c, width)
    points_a = np.array([points[u] - centre_rot[u] for u in range(dim)])
    vect = space.tangent_bundle.element(Rot_inf(points_a.T).T)

    return vect*I



def generate_image_2articulations_vectfield_1(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of the top part of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width
    used to generate vector field in generate_vectorfield_2articulations_1
    """
    dim=2
    points=space.points().T
    limit = 0.3*width

    vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(b, c)
    base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
    base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]

    base_points_b = []
    base_points_b.append(cmp.solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
    base_points_b.append(cmp.solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())

    limit_ab  = 0.2*ab_norm
    limit_bc  = 0.2*bc_norm
    limit_orth = 0.2*width



    points_prod_0_ab = sum([(points[u] - base_points_b[0][u])*vector_ab_unit[u] for u in range(dim)])
    points_prod_0_bc = sum([(points[u] - base_points_b[0][u])*vector_bc_unit[u]  for u in range(dim)])

    points_prod_0_ab_orth = sum([(points[u] - base_points_b[0][u])*vector_ab_norm_orth[u] for u in range(dim)])
    points_prod_0_bc_orth = sum([(points[u] - base_points_b[0][u])*vector_bc_norm_orth[u]  for u in range(dim)])


    I_arti0 = (-ab_norm - limit_ab - limit<= points_prod_0_ab )*(points_prod_0_ab <= 0 + 0*limit_ab + limit)
    I_arti0 *= (0 - limit <= points_prod_0_ab_orth )*(points_prod_0_ab_orth <= width + 0*limit_orth + limit)



    I_arti1 = (0 - 0*limit<= points_prod_0_bc )*(points_prod_0_bc <= bc_norm + limit_bc + limit)
    I_arti1 *= (0 - limit <= points_prod_0_bc_orth )*(points_prod_0_bc_orth <= width + 0*limit_orth + limit)

    return space.element( (I_arti1 == 1))
#
#


def generate_vectorfield_2articulations_1(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a vector field corresponding to rotation centred on the 'finger
    ' with 2 articulations at a and b, with ending point at c,
    rotating the top part of the finger (2nd 'articulation)
    """

    dim = 2
    vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(b, c)
    base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
    base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]

    points=space.points().T
    limit = 0.3*width

    base_points_b = []
    base_points_b.append(cmp.solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
    base_points_b.append(cmp.solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())
    points_prod_0_bc_orth = space.element(sum([(points[u] - base_points_b[0][u])*vector_bc_norm_orth[u]  for u in range(dim)]))

    centre_rot0 = base_points_b[0].copy()
    centre_rot1 = base_points_b[1].copy()

    points = space.points().T
    I = generate_image_2articulations_vectfield_1(space, a, b, c, width)
    points_dec0 = np.array([points[u] - centre_rot0[u] for u in range(dim)])
    fac0 = ((points_prod_0_bc_orth - (width + limit))**2)**(1/2)
    vect0 = fac0*space.tangent_bundle.element(Rot_inf(points_dec0.T).T)
    fac1 = (( (points_prod_0_bc_orth - ( - limit)))**2)**(1/2)
    points_dec1 = np.array([points[u] - centre_rot1[u] for u in range(dim)])
    vect1 = fac1*space.tangent_bundle.element(Rot_inf(points_dec1.T).T)

    return ((vect0 + vect1) / (fac0 + fac1))*I
