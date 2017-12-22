#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:56:16 2017

@author: bgris
"""


import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tl

import sys
sys.path.insert(0, '/home/bgris')
#deform/FromPointsVectorsCoeff.py

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

import scipy

import random as rd
from DeformationModulesODL.deform import Kernel

#%%


space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[48, 48],
        dtype='float32', interp='linear')


width = 1

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

    v_temp = gen.generate_vectorfield_2articulations_0(space, param[0:2], param[2:4], param[4:6], width).copy()
    truth_temp = np.empty((48, 48, 2))
    truth_temp[:, :, 0] = v_temp[0].copy()
    truth_temp[:, :, 1] = v_temp[1].copy()

    return truth_temp.copy()
#

#a = [0, 0]
#theta_b = 0.5*np.pi
#theta_c = 0.5*np.pi
#r_b = 2
#r_c = 5
#a, b, c = generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c)
#gen.generate_vectorfield_2articulations_0(space, a, b, c, width).show()


r_b = 5
r_c = 5
sigma = 0.3

nbdata = 10
param =  generate_random_param(nbdata, r_b, r_c)
points_list = []
vectors_list = []
vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(param.T[0][0:2], param.T[0][2:4])
vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(param.T[0][2:4], param.T[0][4:6])
nb_ab = int((ab_norm + 0.2*width) / sigma) +1
nb_ab_orth = int(2 * width / sigma) +1
nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
nb_bc_orth = int(2*width / sigma) +1

truth =[]

for i in range(nbdata):
    a = param.T[i][0:2]
    b = param.T[i][2:4]
    c = param.T[i][4:6]
    truth_temp = generate_truth_from_param(param.T[i]).copy()
    truth.append(truth_temp.copy())
    points, vectors = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
    points_list.append(points.copy())
    vectors_list.append(vectors.copy())



#%% Graph
inp0 = tf.placeholder(shape=(None,6), dtype=tf.float32)
inp1 = tf.placeholder(shape=(None, n_tot), dtype=tf.float32)
tf.contrib.image.transform
f_layer = tl.fully_connected(inp, num_outputs=4096)
reshaped = tf.reshape(f_layer, (-1, 4, 4, 256))
conv1 = tl.conv2d_transpose(reshaped, num_outputs=128, kernel_size=5, stride=2)
conv2 = tl.conv2d_transpose(conv1, num_outputs=64, kernel_size=5, stride=3)
output = tl.conv2d_transpose(conv2, num_outputs=2, kernel_size=5, stride=2)
#output = tf.contrib.layers.fully_connected(mid_layer, num_outputs=25*2, activation_fn=None)
vel = tf.placeholder(shape=(None, 48, 48, 2), dtype=tf.float32)
loss = tf.norm(vel - output)



#%% Graph
inp = tf.placeholder(shape=(None,6), dtype=tf.float32)
tf.contrib.image.transform
f_layer = tl.fully_connected(inp, num_outputs=4096)
reshaped = tf.reshape(f_layer, (-1, 4, 4, 256))
conv1 = tl.conv2d_transpose(reshaped, num_outputs=128, kernel_size=5, stride=2)
conv2 = tl.conv2d_transpose(conv1, num_outputs=64, kernel_size=5, stride=3)
output = tl.conv2d_transpose(conv2, num_outputs=2, kernel_size=5, stride=2)
#output = tf.contrib.layers.fully_connected(mid_layer, num_outputs=25*2, activation_fn=None)
vel = tf.placeholder(shape=(None, 48, 48, 2), dtype=tf.float32)
loss = tf.norm(vel - output)

#%% Session

r_b = 5
r_c = 5


#%%
param0 = generate_random_param(1, r_b, r_c)

truth0 =[generate_truth_from_param(param0)]
space.element(truth0[0][:, :, 0]).show()
space.element(truth0[0][:, :, 1]).show()
param0
#%%