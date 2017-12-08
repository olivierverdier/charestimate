#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:21:25 2017

@author: bgris
"""

import structured_vector_fields
import group
import pytest
import numpy as np
import odl
import matplotlib.pyplot as plt
import estimate_structured_base_pointsvectors as est_coeff
import function_compute_pointsvectors as cmp
import generate_data_doigt as gen
import structured_vector_fields as struct


a = [0, 0]
b = [0, 2]
c = [-2, 5]
mini = -5
maxi = 10

width = 1
sigma = 0.1

#points, vectors = cmp.compute_pointsvectors_2articulations(a, b, c, width, sigma)

#plt.figure()
#plt.plot(a[0], a[1], 'or')
#plt.plot(b[0], b[1], 'or')
#plt.plot(c[0], c[1], 'or')
#plt.plot(points_list[0][0], points_list[0][1], 'xb')
#plt.axis([mini, maxi, mini, maxi]), plt.grid(True, linestyle='--')


#%%



#%% generate data


space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[128, 128],
        dtype='float32', interp='linear')


width = 1
vector_fields_list = []
vector_fields_list_proj=[]
image_list = []
#a_list = [[0,0], [0,0], [0,0], [0, 0]]
#b_list = [[0, 2], [0, 2], [-2, 0], [2*np.cos(np.pi*0.25), 2*np.sin(np.pi*0.25)]]
#c_list = [[0, 5], [-5, 2], [-5, 0], []]

r_b = 2
theta_list_b = [np.pi*0.5 , np.pi*0.5 , np.pi*0.75, np.pi*0.75, np.pi*0.75, np.pi ]

r_c = 5
theta_list_c = [0 , np.pi*0.25, 0 , np.pi*0.25, np.pi*0.5, 0 ]

nbdata = 6
a_list=[]
b_list=[]
c_list=[]

for i in range(nbdata):
    a_list.append([0., 0.])
    b_list.append([a_list[i][0] + r_b*np.cos(theta_list_b[i]), a_list[i][1] + r_b*np.sin(theta_list_b[i])])
    c_list.append([b_list[i][0] + r_c*np.cos(theta_list_c[i] +theta_list_b[i]), b_list[i][1] + r_c*np.sin(theta_list_c[i] +theta_list_b[i])])

points_list = []
vectors_list = []
vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(a, b)
vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(b, c)
nb_ab = int(ab_norm / sigma) +1
nb_ab_orth = int(width / sigma) +1
nb_bc = int(bc_norm / sigma)
nb_bc_orth = int(width / sigma)

for i in range(nbdata):
    a = a_list[i]
    b = b_list[i]
    c = c_list[i]
    vector_fields_list.append(gen.generate_vectorfield_2articulations_0(space, a, b, c, width))

    image_list.append(gen.generate_image_2articulations(space, a, b, c, width))
    points, vectors = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
    points_list.append(points.copy())
    vectors_list.append(vectors.copy())

#

if False:
    for i in range(nbdata):
        image_list[i].show()
#
#%% Compute coeff

def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x,y)]) / (sigma ** 2))


alpha = est_coeff.estimate_linear_coeff(kernel, vector_fields_list, points_list, vectors_list)

#%% see difference computed vs real vector fields
dim = 2
nb_vectors = len(vectors_list[0][0])
nb_points = len(points_list[0][0])
gen_unstructured = struct.get_from_structured_to_unstructured(space, kernel)

for i in range(nbdata):
    a = a_list[i]
    b = b_list[i]
    c = c_list[i]
    points, vectors = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
    vector_translations = np.array([sum([alpha[u::nb_vectors]*vectors[v, u] for u in range(nb_vectors)]) for v in range(dim)])
    structured_i = struct.create_structured(points, vector_translations)
    unstructured_i = gen_unstructured(structured_i)
    (unstructured_i - vector_fields_list[i]).show('difference '+str(i))
    (unstructured_i).show('computed '+str(i))
    ( vector_fields_list[i]).show('true '+str(i))
#

#%% test compute new field

#
#a_test = [0, 0]
#b_test = [-2, 0]
#c_test = [-2, -5]

theta_b_test = 1.3*np.pi
theta_c_test = 0.6*np.pi

a_test =[0., 0.]
b_test = [a_test[0] + r_b*np.cos(theta_b_test), a_test[1] + r_b*np.sin(theta_b_test)]
c_test = [b_test[0] + r_c*np.cos(theta_c_test + theta_b_test), b_test[1] + r_c*np.sin(theta_c_test + theta_b_test)]
points, vectors = cmp.compute_pointsvectors_2articulations_nb(a_test, b_test, c_test, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
vector_translations = np.array([sum([alpha[u::nb_vectors]*vectors[v, u] for u in range(nb_vectors)]) for v in range(dim)])
structured_test = struct.create_structured(points, vector_translations)
unstructured_test = gen_unstructured(structured_test)
unstructured_test.show(clim=[-2, 2])








