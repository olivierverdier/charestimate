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

import numpy as np
import odl
import matplotlib.pyplot as plt
import estimate_structured_base_pointsvectors as est_coeff
import function_compute_pointsvectors as cmp
#import generate_data_doigt as gen
import structured_vector_fields as struct

import scipy

import random as rd
#from DeformationModulesODL.deform import Kernel


#%%


space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[512, 512],
        dtype='float32', interp='linear')


width = 1

#a = [0, 0]
#theta_b = 0.5*np.pi
#theta_c = 0.5*np.pi
#r_b = 2
#r_c = 5
#a, b, c = generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c)
#gen.generate_vectorfield_2articulations_0(space, a, b, c, width).show()


r_b = 2
r_c = 4
sigma = 1
sblur = 5
nbdata = 10
#param =  gen.generate_random_param(nbdata, r_b, r_c)
#points_list = []
#vectors_list = []
#vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(param.T[0][0:2], param.T[0][2:4])
#vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(param.T[0][2:4], param.T[0][4:6])
#nb_ab = int((ab_norm + 0.2*width) / sigma) +1
#nb_ab_orth = int(2 * width / sigma) +1
#nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
#nb_bc_orth = int(2*width / sigma) +1
#
#truth =[]
#
#for i in range(nbdata):
#    a = param.T[i][0:2]
#    b = param.T[i][2:4]
#    c = param.T[i][4:6]
#    truth_temp = gen.generate_truth_from_param(param.T[i]).copy()
#    truth.append(truth_temp.copy())
#    points, vectors = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
#    points_list.append(points.copy())
#    vectors_list.append(vectors.copy())
##
#
#n_tot = (nb_ab + nb_ab_orth + nb_bc + nb_bc_orth) * 2
#nb_vectors = len(vectors_list[0][0])
#nb_points = len(points_list[0][0])
mg = space.meshgrid
dim = 2

#%% Useful functions
"""
We supposed that data are given as a list of Nb_data structured vector fields

width, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth, sigma, dim are fixed
"""

#tf.contrib.image.transform
def kernel(x, y):
    #si = tf.shape(x)[0]
    return tf.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(dim)]) / (sigma ** 2))




def make_covariance_matrix(points1, points2):
    """ creates the covariance matrix of the kernel for the given points"""

    #dim = tf.shape(points)[0]
    p1 = tf.reshape(points1, (dim, 1, -1))
    p2 = tf.reshape(points2, (dim, -1, 1))

    return kernel(p1, p2)

def squared_norm_diff(structured1, structured2):

    points1 = structured1[0:dim]
    vectors1 = structured1[dim:2*dim]
    points2 = structured2[0:dim]
    vectors2 = structured2[dim:2*dim]

    cov1 = make_covariance_matrix(points1, points1)
    cov12 = make_covariance_matrix(points1, points2)
    cov2 = make_covariance_matrix(points2, points2)

    norm1 = sum([tf.matmul([vectors1[u]], tf.matmul(cov1, tf.transpose([vectors1[u]]))) for u in range(dim)])
    prod_scal12 = sum([tf.matmul([vectors1[u]], tf.matmul(cov12, tf.transpose([vectors2[u]]))) for u in range(dim)])
    norm2 = sum([tf.matmul([vectors2[u]], tf.matmul(cov2, tf.transpose([vectors2[u]]))) for u in range(dim)])

    return norm1 - 2 * prod_scal12 + norm2



def compute_vect_unit(a, b):
    """
    computes unit vectors colin to ab and its orthogonal """
    vector_ab = tf.constant([b[u] - a[u] for u in range(dim)])
    norm_ab = tf.sqrt(sum( [vector_ab[u]**2 for u in range(dim)]))
    vector_ab_unit = vector_ab / norm_ab
    vector_ab_norm_orth = [-vector_ab[1], vector_ab[0]]/norm_ab
    return [vector_ab_unit, vector_ab_norm_orth, norm_ab]



def scalar_product_structured(structured1, structured2):
    points1 = structured1[0:dim]
    vectors1 = structured1[dim:2*dim]
    points2 = structured2[0:dim]
    vectors2 = structured2[dim:2*dim]

    scalar_product = sum([kernel(points1[:, i], points2[:, j]) * tf.tensordot(vectors1[:, i], vectors2[:, j], 1) for i in range(nb_points) for j in range(nb_points)])

    return scalar_product



def scalar_product_structured_cov_mat(structured1, structured2, CovMat):
    vectors1 = structured1[dim:2*dim]
    vectors2 = structured2[dim:2*dim]

    #    scalar_product = sum([CovMat[i, j] * tf.tensordot(vectors1[:, i], vectors2[:, j], 1) for i in range(nb_points) for j in range(nb_points)])

    scalar_product = sum([ tf.tensordot(vectors1[u], tf.tensordot( CovMat, tf.transpose(vectors2[u]), 1), 1) for u in range(dim)])

    return scalar_product



def create_structured(points, vectors):
    return tf.concat([points, vectors], 0)


def mult_scalar_structured(scal, structured):
    points = structured[0:dim]
    vectors = structured[dim:2*dim]
    vectors_mult = scal * vectors
    return create_structured(points, vectors_mult)


def mult_scalar_structured_np(scal, structured):
    points = structured[0:dim]
    vectors = structured[dim:2*dim]
    vectors_mult = scal * vectors
    return struct.create_structured(points, vectors_mult)



#
#f_layer = tl.fully_connected(inp, num_outputs=4096)
#reshaped = tf.reshape(f_layer, (-1, 4, 4, 256))
#conv1 = tl.conv2d_transpose(reshaped, num_outputs=128, kernel_size=5, stride=2)
#conv2 = tl.conv2d_transpose(conv1, num_outputs=64, kernel_size=5, stride=3)
#output = tl.conv2d_transpose(conv2, num_outputs=2, kernel_size=5, stride=2)
##output = tf.contrib.layers.fully_connected(mid_layer, num_outputs=25*2, activation_fn=None)
#vel = tf.placeholder(shape=(None, 48, 48, 2), dtype=tf.float32)
#loss = tf.norm(vel - output)
#

#%% Load data
def kernel_np(x, y):
    #si = tf.shape(x)[0]
    return np.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(dim)]) / (sigma ** 2))



#path = '/home/bgris/data/RotationRectangle/'
#path = '/home/bgris/data/Doigtbis/'
path = '/home/bgris/data/Doigtbis_dimcont2_thin/'
#name_exp = 'rb_' + str(r_b) +  '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
#name_exp = 'rb_' + str(r_b) + '_width_' + str(width) + '_sigma_' + str(sigma) + 'nb_fixed' + '_nbdata_' + str(nbdata)
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) +   '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata) + '_sblur_' + str(sblur)

name = path + name_exp + '/'

structured_list = []
unstructured_list = []
points_list = []
vectors_list = []
cov_mat_list = []
param_list = []
A_inner_prod_list = []

nbdatamax = 10
for i in range(nbdatamax):
    structured_list.append(np.loadtxt(name + 'structured' + str(i)))
    unstructured_list.append(np.loadtxt(name + 'unstructured' + str(i)))
    vectors_i = structured_list[i][dim:2*dim]
    points_list.append(np.loadtxt(name + 'points' + str(i)))
    vectors_list.append(np.loadtxt(name + 'vectors' + str(i)))
    param_list.append(np.loadtxt(name + 'param' + str(i)))
    cov_mat_list.append(struct.make_covariance_matrix(points_list[i], kernel_np))
    A_inner_prod_list.append(np.dot(cov_mat_list[i], np.dot(vectors_i.T, vectors_list[i] )).T)
#

param_list = np.array(param_list).T

nb_points = len(points_list[0][0])
nb_vectors = len(vectors_list[0][0])
##%% Graph 1 : projection of data on zeta(o, 1) and diff with data
#
##inp = tf.placeholder(shape=(nb_vectors, nb_points), dtype=tf.float64)
#inp = tf.Variable(np.ones([nb_vectors, nb_points]), name="alpha")
#
## List of zeta_(o_i) (1)
#structured_list_computed = [create_structured(points_list[i], tf.matmul(vectors_list[i], inp)) for i in range(nbdatamax)]
#
##squared_norm_list_computed = [scalar_product_structured(structured_list_computed[i], structured_list_computed[i]) for i in range(nbdatamax)]
#
#scalar_product_list = [scalar_product_structured_cov_mat(structured_list[i], structured_list_computed[i], cov_mat_list[i]) / scalar_product_structured(structured_list_computed[i], structured_list_computed[i])  for i in range(nbdatamax)]
#output = sum([squared_norm_diff(tf.constant(structured_list[i]), mult_scalar_structured(scalar_product_list[i], structured_list_computed[i])) for i in range(nbdatamax)])
#
#
##%% Graph 2 : projection of zeta(o, 1) on data and diff with zeta(o, 1)
#norm_list = [struct.scalar_product_structured(structured_list[i], structured_list[i], kernel_np) for i in range(nbdatamax)]
#unit_vectfieldlist = [mult_scalar_structured(1 / np.sqrt(norm_list[i]), structured_list[i]) for i in range(nbdatamax)]
#
##inp = tf.placeholder(shape=(nb_vectors, nb_points), dtype=tf.float64)
#inp = tf.Variable(np.ones([nb_vectors, nb_points]), name="alpha")
#
## List of zeta_(o_i) (1)
#structured_list_computed = [create_structured(points_list[i], tf.matmul(vectors_list[i], inp)) for i in range(nbdatamax)]
#
##squared_norm_list_computed = [scalar_product_structured(structured_list_computed[i], structured_list_computed[i]) for i in range(nbdatamax)]
#
#scalar_product_list = [scalar_product_structured(unit_vectfieldlist[i], structured_list_computed[i])  for i in range(nbdatamax)]
#output = sum([squared_norm_diff(structured_list_computed[i], mult_scalar_structured(scalar_product_list[i], tf.constant(unit_vectfieldlist[i]))) for i in range(nbdatamax)])
#
#
##%% Graph 3 : projection of zeta(o, 1) on data and diff with zeta(o, 1)
## inner product using A_inner_prod_list
#squared_norm_list = [struct.scalar_product_structured(structured_list[i], structured_list[i], kernel_np) for i in range(nbdatamax)]
#vectfield_dividedsquarednorm_list = [mult_scalar_structured_np(1 / squared_norm_list[i], structured_list[i]) for i in range(nbdatamax)]
#
##inp = tf.placeholder(shape=(nb_vectors, nb_points), dtype=tf.float64)
#inp = tf.Variable(np.ones([nb_vectors, nb_points]), name="alpha")
#
## List of zeta_(o_i) (1)
#structured_list_computed = [create_structured(points_list[i], tf.matmul(vectors_list[i], inp)) for i in range(nbdatamax)]
#
##squared_norm_list_computed = [scalar_product_structured(structured_list_computed[i], structured_list_computed[i]) for i in range(nbdatamax)]
#
## scalar products between data and generated : needs to
#scalar_product_list = [tf.reduce_sum(tf.multiply(A_inner_prod_list[i], inp ))  for i in range(nbdatamax)]
#output = sum([squared_norm_diff(structured_list_computed[i], mult_scalar_structured(scalar_product_list[i], tf.constant(vectfield_dividedsquarednorm_list[i]))) for i in range(nbdatamax)])
#
#

#%% Graph 4 : projection of data on zeta(o, 1) and diff with data
# inner product using A_inner_prod_list
# simplified norm

vectors_product_spaces = [np.array([[np.dot(vectors_list[i].T[j], vectors_list[i].T[k]) for j in range(nb_vectors)] for k in range(nb_vectors)]) for i in range(nbdatamax)]

#inp = tf.placeholder(shape=(nb_vectors, nb_points), dtype=tf.float64)
inp = tf.Variable(np.ones([nb_vectors, nb_points]), name="alpha")

# List of zeta_(o_i) (1)
structured_list_computed = [create_structured(points_list[i], tf.matmul(vectors_list[i], inp)) for i in range(nbdatamax)]

#squared_norm_list_computed = [scalar_product_structured(structured_list_computed[i], structured_list_computed[i]) for i in range(nbdatamax)]

# scalar products between data and generated : needs to
scalar_product_list = [tf.reduce_sum(tf.multiply(A_inner_prod_list[i], inp ))  for i in range(nbdatamax)]
squared_norm_list = [sum([vectors_product_spaces[i][j][k] * tf.matmul(tf.reshape(inp[k], [1, nb_points]), tf.matmul(cov_mat_list[i], tf.transpose([inp[j]]))) for k in range(nb_vectors) for j in range(nb_vectors)]) for i in range(nbdatamax)]

output = sum([squared_norm_diff(tf.constant(structured_list[i]), mult_scalar_structured(scalar_product_list[i] / squared_norm_list[i], structured_list_computed[i])) for i in range(nbdatamax)])


##%%
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#energy_init = sess.run(output)
#learning_rate = 0.001
#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(output)

##%%
#sess.run(optimizer)
#energy_opt = sess.run(output)
#print('optimized energy = {}'.format(energy_opt))
#%% Gradient descent initialisation of session

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
grads = tf.gradients(output, [inp])
grad = grads[0]
#%% Gradient descent initialisation
nb_it = 10000

alpha_init = np.ones([nb_vectors, nb_points])
alpha_init[0][0] = 1.0
alpha = alpha_init.copy()

##%% Gradient descent run
energy = output.eval(feed_dict={inp: alpha_init})

print('init  energy = {}'.format( energy))
step = 0.2
#%%
for i in range(nb_it):
    energy_grad = grad.eval(feed_dict={inp: alpha})
    alpha_temp = alpha - step * energy_grad
    energy_temp = output.eval(feed_dict={inp: alpha_temp})
    if (energy_temp < energy):
        print('i = {}  energy = {},    step = {}'.format(i, energy, step))
        step *= 1.2
        alpha = alpha_temp.copy()
        energy = energy_temp.copy()
    else:
        step *=0.8
        print('i = {}   step = {}'.format(i, step))
#


#%%

#path = '/home/bgris/Results/DeformationModules/RotationRectangle/'
path = '/home/bgris/Results/DeformationModule/Doigtbisthin/'
##name_exp = 'rb_' + str(r_b) + '_rc_' +  '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
#name_exp = 'rb_' + str(r_b) + '_width_' + str(width) + '_sigma_' + str(sigma) + 'nb_fixed' + '_nbdata_' + str(nbdata)

name = path + name_exp

np.savetxt(name + 'alpha', alpha)

##%%
#learning_rate = 0.1
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(output, var_list=[inp])
#
#
#tf.train.GradientDescentOptimizer.minimize(
#    output,
#    global_step=None,
#    var_list=[inp],
#    gate_gradients=grad,
#    aggregation_method=None,
#    colocate_gradients_with_ops=False,
#    name=None,
#    grad_loss=None
#)

#%%% Visualize results
def kernel_np(x, y):
    #si = tf.shape(x)[0]
    return np.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(dim)]) / (sigma ** 2))

get_unstructured_op = struct.get_from_structured_to_unstructured(space, kernel_np)
structured_computed = []
unstructured_computed = []
fac = []

for i in range(nbdatamax):
    structured_temp = struct.create_structured(points_list[i], np.dot(vectors_list[i], alpha))
    prod = struct.scalar_product_structured(structured_temp, structured_list[i], kernel_np)
    squared_norm = struct.scalar_product_structured(structured_temp, structured_temp, kernel_np)
    fac.append(prod / squared_norm)
    structured_computed.append(structured_temp.copy())
    unstructured_computed.append(get_unstructured_op(structured_temp).copy())
#



for i in range(nbdatamax):
    get_unstructured_op(structured_list[i]).show('unstructured data {}'.format(i))
    (fac[i] * unstructured_computed[i]).show('unstructured computed {}'.format(i))
#



for i in range(nbdatamax):
    (fac[i] * unstructured_computed[i] -get_unstructured_op(structured_list[i])).show('Difference {}'.format(i))

#
#%% save result

path = '/home/bgris/Results/DeformationModules/Doigtbis/'
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
name = path + name_exp

np.savetxt(name + 'alpha', alpha)

alpha_load = np.loadtxt(name + 'alpha')


#%%







#%%
#
#inp = np.ones([nb_vectors, nb_points])
#structured_list_computed = [create_structured(points_list[i], tf.matmul(vectors_list[i], inp)) for i in range(nbdatamax)]
#
##squared_norm_list_computed = [scalar_product_structured(structured_list_computed[i], structured_list_computed[i]) for i in range(nbdatamax)]
#
#scalar_product_list = [scalar_product_structured(structured_list[i], structured_list_computed[i]) / scalar_product_structured(structured_list_computed[i], structured_list_computed[i])  for i in range(nbdatamax)]
#output = sum([squared_norm_diff(structured_list[i], mult_scalar_structured(scalar_product_list[i], structured_list_computed[i])) for i in range(nbdatamax)])
#
#


#%%
y = scalar_product_list[0]
i=0
y0 = mult_scalar_structured(scalar_product_list[i], structured_list_computed[i])
y1 = squared_norm_diff(structured_list[i], y0)
#%%% Old one
inp0 = tf.placeholder(shape=(None, 6), dtype=tf.float32)
inp1 = tf.placeholder(shape=(None, n_tot), dtype=tf.float32)
#inp2 = tf.placeholder(shape=(None, nbdata), dtype=tf.float32)

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

structured1 = tf.constant([[0, 1, 2], [0, 0, 0], [2, 0, 1], [0, 3, 4]], dtype = 'float64')



#%%

#sess = tf.Session()
#sess.run(y)
r_b = 5
r_c = 5


#%%
param0 = generate_random_param(1, r_b, r_c)

truth0 =[generate_truth_from_param(param0)]
space.element(truth0[0][:, :, 0]).show()
space.element(truth0[0][:, :, 1]).show()
param0
#%%