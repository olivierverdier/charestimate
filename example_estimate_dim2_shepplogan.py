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


#%% Get data

# Size of dataset
size = 10
space=odl.uniform_discr(
min_pt=[-16, -16], max_pt=[16, 16], shape=[512,512],
dtype='float32', interp='linear')
data_list=[]
path='/home/bgris/data/SheppLoganRotationSmallDef/vectfield512/'
name='vectfield_smalldef_sigma_1'
for i in range(size):
    name_i=path + name + '_{}'.format(i)
    vect_field_load_i_test=space.tangent_bundle.element(np.loadtxt(name_i)).copy()
    data_list.append(vect_field_load_i_test.copy())
#
#%% Set parameters kernel and control points

# scale of the kernel
sigma_kernel=1
fac=0.5
xmin=-5
xmax=5
dx=round((xmax-xmin)/(fac*sigma_kernel))
ymin=-5.0
ymax=5.0
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

# define calibration
get_unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)

# define calibration
fun_op = func_2D.function_2D_scalingdisplacement
def calibration_equation(original, noisy):
    result = cali.calibrate_equation(original, noisy, space, kernel, fun_op)
    return result

calibration = calibration_equation

#%%



sigma0 = 1
sigma1 = 500

dim = 1
nb_iteration = 10
points = np.array(points_list).T
# first raw estimation
result = scheme.iterative_scheme(solve_regression, calibration_equation, action, g,
                                 kernel, data_list, sigma0,
                                 sigma1, points, nb_iteration)
#
#%% Compare result with all data
result_unstruc = get_unstructured_op(result[0])
result_unstruc.show('computed', clim = [-0.1, 0.1])

for i in range(size):
    velo = calibration_equation(result[0], data_list[i])
    computed = action(g.exponential(velo), result[0])
    result_unstruc_i = get_unstructured_op(computed)
    (data_list[i]).show('data {}'.format(i), clim = [-0.1, 0.1])
    result_unstruc_i.show('computed calibrated {}'.format(i), clim = [-0.1, 0.1])
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
#%% Save results
name = '/home/bgris/DeformationModulesODL/deform/vect_field_rotation_SheppLogan_scheme_equation_sigma_1__nbtrans_441_nbiteration_5'
np.savetxt(name,result_unstruc)


vec = space.tangent_bundle.element(np.loadtxt(name))



