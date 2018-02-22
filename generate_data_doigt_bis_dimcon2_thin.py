#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:45:40 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:41:46 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:11:03 2018

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
import function_generate_data_doigt_bis as fun_gen
import group
import pytest
import numpy as np
import odl
import estimate_structured_base_pointsvectors as est_coeff
import os
import matplotlib.pyplot as plt
import numpy.random as rd
import scipy.ndimage as ndimage
#

dim = 2

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

def generate_truth_from_param(param, Cont0, Cont1):

    v_temp0 = fun_gen.generate_vectorfield_2articulations_0(space, param[0:2], param[2:4], param[4:6], width).copy()
    v_temp1 = fun_gen.generate_vectorfield_2articulations_1(space, param[0:2], param[2:4], param[4:6], width).copy()
    truth_temp = np.empty((space.shape[0], space.shape[1], 2))
    truth_temp[:, :, 0] = Cont0 *v_temp0[0] + Cont1 * v_temp1[0]
    truth_temp[:, :, 1] = Cont0 *v_temp0[1] + Cont1 * v_temp1[1]

    return truth_temp.copy()
#

def generate_truthblurred_from_param(param, Cont0, Cont1, sblur):

    dim = 2
    v_temp0 = fun_gen.generate_vectorfield_2articulations_0(space, param[0:2], param[2:4], param[4:6], width).copy()
    v_temp0 = [ndimage.gaussian_filter(v_temp0[u], sigma = (sblur,sblur),  order=0) for u in range(dim)]
    v_temp0 = space.tangent_bundle.element(v_temp0)
    v_temp1 = fun_gen.generate_vectorfield_2articulations_1(space, param[0:2], param[2:4], param[4:6], width).copy()
    v_temp1 = [ndimage.gaussian_filter(v_temp1[u], sigma = (sblur,sblur),  order=0) for u in range(dim)]
    v_temp1 = space.tangent_bundle.element(v_temp1)
    truth_temp = np.empty((space.shape[0], space.shape[1], 2))
    truth_temp[:, :, 0] = Cont0 *v_temp0[0] + Cont1 * v_temp1[0]
    truth_temp[:, :, 1] = Cont0 *v_temp0[1] + Cont1 * v_temp1[1]

    return truth_temp.copy()
#


#%% generate data

space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[512, 512],
        dtype='float32', interp='linear')


vector_fields_list = []
image_list = []


r_b = 2
r_c = 4
sigma = 1
width = 0.3 * sigma

nbdata = 100
param =  generate_random_param(nbdata, r_b, r_c)
points_list = []
vectors_list = []

a= param.T[0][0:2]
b= param.T[0][2:4]
c= param.T[0][4:6]

#vector_ab_unit, vector_ab_norm_orth, ab_norm = cmp.compute_vect_unit(a, b)
#vector_bc_unit, vector_bc_norm_orth, bc_norm = cmp.compute_vect_unit(b, c)
#
#base_points_a = [a - 0.5*width *vector_ab_norm_orth, a + 0.5*width *vector_ab_norm_orth ]
#base_points_c = [c - 0.5*width *vector_bc_norm_orth, c + 0.5*width *vector_bc_norm_orth ]
#
#base_points_b = []
#base_points_b.append(cmp.solve_intersection(base_points_a[0], vector_ab_unit, base_points_c[0], vector_bc_unit, ab_norm).copy())
#base_points_b.append(cmp.solve_intersection(base_points_a[1], vector_ab_unit, base_points_c[1], vector_bc_unit, ab_norm).copy())
#
#
#vector_points_b_unit, vector_points_b_norm_orth, points_b_norm = cmp.compute_vect_unit(base_points_b[0], base_points_b[1])

#n_orth = int((points_b_norm + 0.2*width) / sigma) +1
nb_ab = 4
nb_bc = 8


Cont = rd.uniform(-2, 2, [nbdata, 2])

sblur = 5

for i in range(nbdata):
    a = param.T[i][0:2]
    b = param.T[i][2:4]
    c = param.T[i][4:6]
    truth_temp = generate_truthblurred_from_param(param.T[i], Cont[i,0], Cont[i,1], sblur).copy()
    #truth_temp = space.tangent_bundle.element(truth_temp)

    image_list.append(fun_gen.generate_image_2articulations(space, a, b, c, width))

    vector_fields_list.append(truth_temp.copy())
    points, vectors = cmp.compute_pointsvectors_2articulations_thin(a, b, c, nb_ab, nb_bc)
    points_list.append(points.copy())
    vectors_list.append(vectors.copy())
#

if False:
    a = param.T[i][0:2]
    b = param.T[i][2:4]
    c = param.T[i][4:6]
    v_temp0 = fun_gen.generate_vectorfield_2articulations_0(space, a, b, c, width).copy()
    v_temp1 = fun_gen.generate_vectorfield_2articulations_1(space, a, b, c, width).copy()
    v_temp1.show()
    plt.plot(param.T[i][::2], param.T[i][1::2])
#

step = 50
points = space.points()
if False:
    i = 2
    fig = image_list[i].show()
    v = space.tangent_bundle.element([vector_fields_list[i][:,:,u] for u in range(2)])
    plt.quiver(points.T[0][::step],points.T[1][::step],v[0][::step],v[1][::step], color='r')
    plt.plot(param.T[i][::2], param.T[i][1::2], 'xb')
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
path = '/home/bgris/data/Doigtbis_dimcont2_thin/'
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata) + '_sblur_' + str(sblur) + '/'
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
np.savetxt(name + 'namebc', [nb_bc])
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

#%% See accuracy of projection


for i in range(nbdata):
    space.tangent_bundle.element([vector_fields_list[i][:,:,u] for u in range(2)])[0].show('truth' + str(i), clim=[-5,5])
#    plt.plot(param.T[i][0::2], param.T[i][1::2], 'xb')
    unstructured_list[i][0].show('projected' + str(i), clim=[-5,5])
    plt.plot(param.T[i][0::2], param.T[i][1::2], 'xb')

#

for i in range(nbdata):
    (space.tangent_bundle.element([vector_fields_list[i][:,:,u] for u in range(2)]) -  unstructured_list[i]).show('diff' + str(i), clim=[-5,5])

#


#%%
for i in range(nbdata):
    image_list[i].show()
    plt.plot(points_list[i][0], points_list[i][1], 'xb')
    plt.plot(param.T[i][0::2], param.T[i][1::2], 'xr')

    space.tangent_bundle.element([vector_fields_list[i][:,:,u] for u in range(2)]).show()
    plt.plot(points_list[i][0], points_list[i][1], 'xb')
    plt.plot(param.T[i][0::2], param.T[i][1::2], 'xr')
#

