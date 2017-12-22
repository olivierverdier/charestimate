#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:15:41 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:58:23 2017

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
#a = [0, 0]
#theta_b = 0.5*np.pi
#theta_c = 0.5*np.pi
#r_b = 2
#r_c = 5
#a, b, c = generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c)
#gen.generate_vectorfield_2articulations_0(space, a, b, c, width).show()


#%% Graph
inp = tf.placeholder(shape=(None, 6), dtype=tf.float32)
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

def generate_truth_from_param(param):

    v_temp = gen.generate_vectorfield_2articulations_0(space, param[0:2], param[2:4], param[4:6], width).copy()
    truth_temp = np.empty((48, 48, 2))
    truth_temp[:, :, 0] = v_temp[0].copy()
    truth_temp[:, :, 1] = v_temp[1].copy()

    return truth_temp.copy()
#
#%%
param0 = generate_random_param(1, r_b, r_c)

truth0 =[generate_truth_from_param(param0)]
space.element(truth0[0][:, :, 0]).show()
space.element(truth0[0][:, :, 1]).show()
param0


#%%

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
#output.eval(feed_dict={inp: np.ones((1,6))}).squeeze().shape
#loss.eval(feed_dict={inp: np.ones((1,6)), vel: np.ones((1, 48, 48, 2))})
#grads = tf.gradients(loss, [inp])
#grad = grads[0]
#grad.eval(feed_dict={inp: np.ones((1,10)), vel: np.ones((1, 48, 48, 2))})
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
mini_step = optimizer.minimize(loss)
session.run(tf.global_variables_initializer())
nb_param_iteration = 10
for i in range(100000):
    param = generate_random_param(nb_param_iteration, r_b, r_c)
    truth =  []
    for k in range(nb_param_iteration):
        truth_temp = generate_truth_from_param(param[:, k]).copy()
        truth.append(truth_temp.copy())
    truth = np.array(truth)
    session.run(mini_step, feed_dict={inp: param.T, vel: truth})
    if (i%1000 == 0):
        print(loss.eval(feed_dict={inp: param0.T, vel: truth0}))
#

#%%

a = [0, 0]
theta_b = 0.5*np.pi
theta_c = 0.5*np.pi
#r_b = 2
#r_c = 5
u = np.array([generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c)])

vect_temp = np.transpose(output.eval(feed_dict={inp: u}).squeeze(), (2, 1, 0))
vect = space.tangent_bundle.element(vect_temp)
vect.show()