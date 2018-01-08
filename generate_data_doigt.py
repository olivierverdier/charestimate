#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:24:15 2017

@author: bgris
"""

import structured_vector_fields as struct
import function_compute_pointsvectors as cmp
import group
import pytest
import numpy as np
import odl
import estimate_structured_base_pointsvectors as est_coeff
import os
import matplotlib.pyplot as plt
import random as rd
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

def generate_image_2articulations(space, a, b, c, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim=2
    points=space.points().T
    limit = 0.0*width

    vector_ab_unit, vector_ab_norm_orth, vector_ab_norm = cmp.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, vector_bc_norm = cmp.compute_vect_unit(b, c)

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

    vector_ab_unit, vector_ab_norm_orth, vector_ab_norm = cmp.compute_vect_unit(a, b)
    vector_bc_unit, vector_bc_norm_orth, vector_bc_norm = cmp.compute_vect_unit(b, c)

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


def generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c):
    b = [a[0] + r_b*np.cos(theta_b), a[1] + r_b*np.sin(theta_b)]
    c = [b[0] + r_c*np.cos(theta_c +theta_b), b[1] + r_c*np.sin(theta_c +theta_b)]

    return np.reshape(np.array([a, b, c]), (-1, 1)).squeeze()
#
def generate_random_param(n, r_b, r_c):
    param = []
    for i in range(n):
        theta_b = rd.uniform(0, 2*np.pi)
        theta_c = rd.uniform(0, 0.5*np.pi)
        a0 = rd.uniform(-1, 1)
        a1 = rd.uniform(-1, 1)
        u =  generate_GD_from_athetas([a0, a1], theta_b, theta_c, r_b, r_c)
        param.append(u.copy())
    return np.array(param).T

def generate_truth_from_param(param):

    v_temp = generate_vectorfield_2articulations_0(space, param[0:2], param[2:4], param[4:6], width).copy()
    truth_temp = np.empty((space.shape[0], space.shape[1], 2))
    truth_temp[:, :, 0] = v_temp[0].copy()
    truth_temp[:, :, 1] = v_temp[1].copy()

    return truth_temp.copy()
#

#%% generate data


space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[128, 128],
        dtype='float32', interp='linear')


width = 1
vector_fields_list = []
image_list = []
#a_list = [[0,0], [0,0], [0,0]]
#b_list = [[0, 2], [0, 2], [-2, 0]]
#c_list = [[0, 5], [-5, 2], [-5, 0]]
#nbdata = 3
#
#for i in range(nbdata):
#    a = a_list[i]
#    b = b_list[i]
#    c = c_list[i]
#    vector_fields_list.append(generate_vectorfield_2articulations_0(space, a, b, c, width))
#    image_list.append(generate_image_2articulations(space, a, b, c, width))
##





r_b = 4
r_c = 4
sigma = 0.5

nbdata = 10
param =  generate_random_param(nbdata, r_b, r_c)
points_list = []
vectors_list = []
vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(param.T[0][0:2], param.T[0][2:4])
vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(param.T[0][2:4], param.T[0][4:6])
nb_ab = int((ab_norm + 0.2*width) / sigma) +1
nb_ab_orth = int(2 * width / sigma) +1
nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
nb_bc_orth = int(2*width / sigma) +1


for i in range(nbdata):
    a = param.T[i][0:2]
    b = param.T[i][2:4]
    c = param.T[i][4:6]
    truth_temp = generate_truth_from_param(param.T[i]).copy()
    image_list.append(generate_image_2articulations(space, a, b, c, width))

    vector_fields_list.append(truth_temp.copy())
    points, vectors = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
    points_list.append(points.copy())
    vectors_list.append(vectors.copy())
#



#%% Create projected vector fields (structured and unstructured)
#sigma = 0.5
#width = 1
## Set number of points
#vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(a_list[0], b_list[0])
#vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(b_list[0], c_list[0])
#nb_ab = int((ab_norm + 0.2*width) / sigma) +1
#nb_ab_orth = int(2 * width / sigma) +1
#nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
#nb_bc_orth = int(2*width / sigma) +1
#


#points_temp, vectors_temp = cmp.compute_pointsvectors_2articulations_nb(a_list[0], b_list[0], c_list[0], width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
nb_vectors = len(vectors_list[0][0])
nb_points = len(points_list[0][0])

dim = 2


# Create list of structured and unstructured
def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma ** 2))

get_unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)
structured_list=[]
unstructured_list=[]

for i in range(nbdata):
    points = points_list[i].copy()
    vectors = vectors_list[i].copy()
    #points, vectors = cmp.compute_pointsvectors_2articulations_nb(a_list[i], b_list[i], c_list[i], width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
    eval_field = np.array([space.element(vector_fields_list[i][:,:,u]).interpolation(
                points) for u in range(dim)]).copy()
    
    vector_syst = np.zeros(dim*nb_points)
    basis = np.identity(dim)

    
    
    for k0 in range(nb_points):
        for l0 in range(dim):
            vector_syst[dim*k0 + l0] += np.dot(eval_field.T[k0],
                    basis[:, l0])
            
    eval_kernel = struct.make_covariance_matrix(points, kernel)

    matrix_syst = np.kron(eval_kernel, basis)

    alpha_concatenated = np.linalg.solve(matrix_syst, vector_syst)
    alpha = struct.get_structured_vectors_from_concatenated(alpha_concatenated, nb_points, dim)
    structured = struct.create_structured(points, alpha)
    
    structured_list.append(structured.copy())
    unstructured_list.append(get_unstructured_op(structured).copy())
#

#%% Save data
path = '/home/barbara/data/Doigt/'
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata) + '/'
name = path + name_exp

os.mkdir(name)
for i in range(nbdata):
    np.savetxt(name + 'structured' + str(i), structured_list[i])
    np.savetxt(name + 'unstructured' + str(i), unstructured_list[i])
    np.savetxt(name + 'param' + str(i), param.T[i])
    np.savetxt(name + 'points' + str(i), points_list[i])
    np.savetxt(name + 'vectors' + str(i), vectors_list[i])
#
np.savetxt(name + 'nameab', [nb_ab])
np.savetxt(name + 'nameaborth', [nb_ab_orth])
np.savetxt(name + 'namebc', [nb_bc])
np.savetxt(name + 'namebc_orth', [nb_bc_orth])
#%% Load data
structured_load = []
unstructured_load = []
points_load = []
vectors_load = []
param_load = []

for i in range(nbdata):
    structured_load.append(np.loadtxt(name + 'structured' + str(i)))
    unstructured_load.append(np.loadtxt(name + 'unstructured' + str(i)))
    points_load.append(np.loadtxt(name + 'points' + str(i)))
    vectors_load.append(np.loadtxt(name + 'vectors' + str(i)))
    param_load.append(np.loadtxt(name + 'param' + str(i)))
#


param_load = np.array(param_load).T

nb_ab_load = int(np.loadtxt(name + 'nameab'))
nb_ab_orth_load = int(np.loadtxt(name + 'nameaborth'))
nb_bc_load = int(np.loadtxt(name + 'namebc'))
nb_bc_orth_load = int(np.loadtxt(name + 'namebc_orth'))
