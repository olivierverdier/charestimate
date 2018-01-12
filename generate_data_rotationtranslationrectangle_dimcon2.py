#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:56:00 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:33:51 2018

@author: bgris
"""

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
import numpy.random as rd
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

def generate_image_rectangle(space, a, b, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim=2
    points=space.points().T

    vector_ab_unit, vector_ab_norm_orth, vector_ab_norm = cmp.compute_vect_unit(a, b)
    #width_list = width * vector_ab_unit
    limit = 0.2*vector_ab_norm
    limit_orth = 0.2*width
    width_list_orth = width * vector_ab_norm_orth

    points_prod_ab = sum([(points[u] - a[u])*vector_ab_unit[u] for u in range(dim)])

    points_prod_ab_orth = sum([(points[u] - a[u] + 0.5*width_list_orth[u])*vector_ab_norm_orth[u] for u in range(dim)])

    I_arti0 = (0-limit <= points_prod_ab )*(points_prod_ab <= vector_ab_norm + limit)
    I_arti0 *= (points_prod_ab_orth >= 0 - limit_orth)* (points_prod_ab_orth <= width + limit_orth)


    return space.element((I_arti0 == 1))
#


def generate_vectorfield_rotationrectangle(space, a, b, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim = 2
    points = space.points().T
    I = generate_image_rectangle(space, a, b, width)
    points_a = np.array([points[u] - a[u] for u in range(dim)])
    vect = space.tangent_bundle.element(Rot_inf(points_a.T).T)

    return vect*I

#

def generate_vectorfield_translationrectangle(space, a, b, width):

    """
    ONLY DIMENSION 2

    generates a black and white image of a 'finger' with 2 articulations
    at a and b, with ending point at c and constant width width
    """

    dim = 2
    I = generate_image_rectangle(space, a, b, width)
    vect = space.tangent_bundle.element([(a[u] - b[u]) * space.one() for u in range(2)])

    return vect*I

#


def generate_GD_from_athetas(a, theta_b, r_b):
    b = [a[0] + r_b*np.cos(theta_b), a[1] + r_b*np.sin(theta_b)]

    return np.reshape(np.array([a, b]), (-1, 1)).squeeze()
#
def generate_random_param(n, r_b):
    param = []
    for i in range(n):
        theta_b = rd.uniform(0, 2*np.pi)
        a0 = rd.uniform(-1, 1)
        a1 = rd.uniform(-1, 1)
        u =  generate_GD_from_athetas([a0, a1], theta_b, r_b)
        param.append(u.copy())
    return np.array(param).T

def generate_truth_from_param(param, cont0, cont1):

    v_temp0 = generate_vectorfield_rotationrectangle(space, param[0:2], param[2:4], width).copy()
    v_temp1 = generate_vectorfield_translationrectangle(space, param[0:2], param[2:4], width).copy()
    truth_temp = np.empty((space.shape[0], space.shape[1], 2))
    truth_temp[:, :, 0] = cont0 * v_temp0[0] + cont1 * v_temp1[0]
    truth_temp[:, :, 1] = cont0 * v_temp0[1] + cont1 * v_temp1[1]

    return truth_temp.copy()
#

#%% generate data


space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[512, 512],
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
sigma = 0.1

nbdata = 10
param =  generate_random_param(nbdata, r_b)
Cont = rd.uniform(-2, 2, [nbdata, 2])
points_list = []
vectors_list = []
vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(param.T[0][0:2], param.T[0][2:4])
nb_ab = int((ab_norm + 0.4*ab_norm) / sigma) +1
nb_ab_orth = int(2 * width / sigma) +1


for i in range(nbdata):
    a = param.T[i][0:2]
    b = param.T[i][2:4]
    truth_temp = generate_truth_from_param(param.T[i], Cont[i][0], Cont[i][1]).copy()
    image_list.append(generate_image_rectangle(space, a, b, width))

    vector_fields_list.append(truth_temp.copy())
    points, vectors = cmp.compute_pointsvectors_rectangle_nb(a, b, 1.2 * width, sigma, nb_ab, nb_ab_orth)
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
#%% See projection

#plt.plot(points[0] , points[1], 'xb')

for i in range(nbdata):
    space.tangent_bundle.element([vector_fields_list[i][:,:,u] for u in range(2)])[0].show('truth' + str(i), clim=[-5,5])
    plt.plot(param.T[i][0::2], param.T[i][1::2], 'xb')
    unstructured_list[i][0].show('projected' + str(i), clim=[-5,5])
    plt.plot(param.T[i][0::2], param.T[i][1::2], 'xb')

#

for i in range(nbdata):
    (space.tangent_bundle.element([vector_fields_list[i][:,:,u] for u in range(2)]) -  unstructured_list[i]).show('diff' + str(i))

#
#%% Save data
path = '/home/bgris/data/RotationTranslationRectangle_dimcont2/'
name_exp = 'rb_' + str(r_b) + '_width_' + str(width) + '_sigma_' + str(sigma) + 'nb_fixed' + '_nbdata_' + str(nbdata) + '/'
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

#%% See decomposition

i = 1

v_temp0 = Cont[i][0] * generate_vectorfield_rotationrectangle(space, param.T[i][0:2], param.T[i][2:4], width).copy()
v_temp1 = Cont[0][1] * generate_vectorfield_translationrectangle(space, param.T[i][0:2], param.T[i][2:4], width).copy()
v_temp0.show(str(i) + 'rotation', clim = [-5, 5])
plt.plot(param.T[i][0],param.T[i][1], 'xr')
v_temp1.show(str(i) + 'translation', clim = [-5, 5])
plt.plot(param.T[i][0],param.T[i][1], 'xr')
plt.plot(points_list[i][0], points_list[i][1], 'xb')