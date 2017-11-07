#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:48:36 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:53:56 2017

@author: bgris
"""


import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.deform.linearized import _linear_deform
from odl.discr import DiscreteLp, Gradient, Divergence
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
import numpy as np
import scipy
import copy
import scheme
import calibration as cali
import regression as reg
import action as act
import group
import accessors as acc
import structured_vector_fields as struct
import functions_calibration_2D as func_2D



#
#%% Set parameters kernel and control points

# scale of the kernel
sigma_kernel=1
fac=0.5
xmin=-6
xmax=6
dx=round((xmax-xmin)/(fac*sigma_kernel))
ymin=-6.0
ymax=6.0
dy=round((ymax-ymin)/(fac*sigma_kernel))
points_list=[]
for i in range(dx+1):
    for j in range(dy+1):
        points_list.append([xmin +fac*sigma_kernel* i*1.0, ymin + fac*sigma_kernel*j*1.0])
#        x0.append(xmin +fac*sigma_kernel* i*1.0)
#        x0.append(ymin + fac*sigma_kernel*j*1.0)
#Number of translations
#nbtrans=round(0.5*len(x0))

def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma_kernel ** 2))

#    scaled = [xi ** 2 / (2 * sigma_kernel ** 2) for xi in x]
#    return np.exp(-sum(scaled))
##
#%% define group parameters and functions

# define group
g = group.ScaleDisplacement

# define regression
solve_regression = reg.solve_regression

def action(group_element, structured_field):
    return act.apply_element_to_field(g, group_element, structured_field)

def product(vect0, vect1):
    return struct.scalar_product_structured(vect0, vect1, kernel)



pairing = struct.scalar_product_unstructured

space=odl.uniform_discr(
min_pt=[-16, -16], max_pt=[16, 16], shape=[128,128],
dtype='float32', interp='linear')

# define calibration
get_unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)

# define calibration
fun_op = func_2D.function_2D_scalingdisplacement
def calibration_equation(original, noisy):
    result = cali.calibrate_equation(original, noisy, space, kernel, fun_op)
    return result

calibration = calibration_equation


get_unstructured_op_generate = get_unstructured_op

#%% Generate data

# Size of dataset
nb_data = 10
points_list = np.array(points_list)

fac=1
xmin=-3
xmax=3
dx=round((xmax-xmin)/(fac*sigma_kernel))
ymin=-3.0
ymax=3.0
dy=round((ymax-ymin)/(fac*sigma_kernel))
points_list_gen=[]
for i in range(dx+1):
    for j in range(dy+1):
        points_list_gen.append([xmin +fac*sigma_kernel* i*1.0, ymin + fac*sigma_kernel*j*1.0])

points_list_gen = np.array(points_list_gen)
nb_pts_gen = len(points_list_gen)
vectors_truth = np.random.uniform(low=-1.0, high=1.0, size = [2, nb_pts_gen])
original = struct.create_structured(points_list_gen.T, vectors_truth)
original_unstructured = get_unstructured_op(original)



data_list = []
nb_data = 10
translation_list = np.random.uniform(low=-1.0, high=1.0, size = [2, nb_data])
scaling_list = np.abs(np.random.normal(1, 1, nb_data)) #1. + np.zeros(nb_data)
theta_list = np.random.uniform(low=-np.pi, high=np.pi, size = nb_data)
param_transfor_list=np.array([ g.exponential(np.array([scaling_list[i], theta_list[i], translation_list[0, i], translation_list[1, i] ]))  for i in range(nb_data)])

#covariance_matrix = struct.make_covariance_matrix(space.points().T, kernel)
#noise_l2 =  odl.phantom.noise.white_noise(odl.ProductSpace(space, nb_data))*0.1
#decomp = np.linalg.cholesky(covariance_matrix + 1e-4 * np.identity(len(covariance_matrix)))
#noise_rkhs = [np.dot(decomp, noise_l2[i]) for i in range(nb_data)]
#pts_space=space.points().T
#data_list=[]
#for i in range(nb_data):
#    pts_displaced = g.apply(np.array([-translation_list[i]]), pts_space)
#    data_list.append(space.tangent_bundle.element([original_unstructured[u].interpolation(pts_displaced) for u in range(dim)]))



#data_list_noisy = [space.tangent_bundle.element(get_unstructured_op_generate(action(np.array(param_transfor_list[i]), original)) + noise_rkhs[i]) for i in range(nb_data)]
data_list = [space.tangent_bundle.element(get_unstructured_op_generate(action(np.array(param_transfor_list[i]), original))) for i in range(nb_data)]




#%%



sigma0 = 1
sigma1 = 500

dim = 1
nb_iteration = 30
points = np.array(points_list).T
# first raw estimation
result = scheme.iterative_scheme(solve_regression, calibration_equation, action, g,
                                 kernel, data_list, sigma0,
                                 sigma1, points, nb_iteration)
#
#%% Compare reult with ground truth

result_unstruc = get_unstructured_op(result[0])
original_unstructured = get_unstructured_op_generate(original)
velo = calibration_equation(result[0], original_unstructured)
computed = action(g.exponential(velo), result[0])
result_unstruc_i = get_unstructured_op(computed)
result_unstruc.show('computed')
original_unstructured.show('ground_truth')
diff_calib = result_unstruc_i - original_unstructured
diff_calib.show('difference after calib')
result_unstruc_i.show('computed calibrated')

#%% Compare result with all data
result_unstruc = get_unstructured_op(result[0])

for i in range(size):
    velo = calibration_equation(result[0], data_list[i])
    computed = action(g.exponential(velo), result[0])
    result_unstruc_i = get_unstructured_op(computed)
    (result_unstruc_i ).show('computed {}'.format(i), clim = [-0.1, 0.1])
    (data_list[i]).show('data {}'.format(i), clim = [-0.1, 0.1])
    ((result_unstruc_i - data_list[i])).show('difference {}'.format(i), clim = [-0.1, 0.1])
    print('iteration ' + str(i))
    print('norm difference ' + str((result_unstruc_i - data_list[i]).norm()))
    print('norm data ' + str(data_list[i].norm()))
#    plt.figure()
#    plt.plot(space.points().T[0], data_list_noisy[i][0].asarray(), label = 'data')
#    plt.plot(space.points().T[0], result_unstruc_i[0].asarray(), label = 'result calibrated')
#    plt.plot(space.points().T[0], result_unstruc[0].asarray(), label = 'result ')
#    plt.legend()
#