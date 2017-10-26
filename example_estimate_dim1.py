#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:44:01 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:57:38 2017

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
# We suppose that we have a list of vector fields vector_field_list

##%%
## Set parameters
# Size of dataset
size = 10


# noise of observation
sigmanoise=0.2

# scale of the kernel
sigma_kernel=0.05
fac=1
xmin=-1
xmax=1 - fac * sigma_kernel
dx=round((xmax-xmin)/(fac*sigma_kernel))
points_list=[]
for i in range(dx+1):
    points_list.append([xmin +fac*sigma_kernel* i*1.0])
#        x0.append(xmin +fac*sigma_kernel* i*1.0)
#        x0.append(ymin + fac*sigma_kernel*j*1.0)

#Number of translations
nbtrans=round(len(points_list))


space=odl.uniform_discr(
min_pt=[-1], max_pt=[1], shape=[128],
dtype='float32', interp='linear')
extent = space.max_pt[0] - space.min_pt[0]

proj = group.projection_periodicity(space)
# to be of the good shape for iterative scheme
points = np.array(points_list).T
sigma_generate = 0.3

def kernel_generate(x, y):
    x_proj = proj(x)
    y_proj = proj(y)
    return np.exp(- sum([ (np.minimum(np.abs(xi - yi), extent -np.abs(xi - yi))) ** 2 for xi, yi in zip(x_proj,y_proj)]) / (sigma_generate ** 2))

def kernel(x, y):
    x_proj = proj(x)
    y_proj = proj(y)
    return np.exp(- sum([ (np.minimum(np.abs(xi - yi), extent -np.abs(xi - yi))) ** 2 for xi, yi in zip(x_proj,y_proj)]) / (sigma_kernel ** 2))

#def kernel_generate(x, y):
#    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)]) / (sigma_generate ** 2))
#
#def kernel(x, y):
#    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x, y)])  / (sigma_kernel ** 2))


#%%

g = group.Translation(space)
solve_regression = reg.solve_regression
calibration = cali.calibrate
def action(group_element, structured_field):
    return act.apply_element_to_field(g, group_element, structured_field)

def product(vect0, vect1):
    return struct.scalar_product_structured(vect0, vect1, kernel)


pairing = struct.scalar_product_unstructured
sigma0 = 1
sigma1 = 10

#%% define calibration
get_unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)

#mg = space.meshgrid
#def get_unstructured_op(structured_field):
#    dim_double, nb_points = structured_field.shape
#    points = struct.get_points(structured_field)
#    vectors = struct.get_vectors(structured_field)
#    unstructured = space.tangent_bundle.zero()
#
#    for k in range(nb_points):
#        def kern_app_point(x):
#            return kernel(proj([xu - pu for xu, pu in zip(x, points[:, k])]) , [0])
#
#        kern_discr = kern_app_point(mg)
#
#        unstructured += space.tangent_bundle.element([kern_discr * vect for vect in vectors[:, k]]).copy()
#
#    return unstructured
#
#def get_unstructured_op_generate(structured_field):
#    dim_double, nb_points = structured_field.shape
#    points = struct.get_points(structured_field)
#    vectors = struct.get_vectors(structured_field)
#    unstructured = space.tangent_bundle.zero()
#
#    for k in range(nb_points):
#        def kern_app_point(x):
#            return kernel_generate(proj([xu - pu for xu, pu in zip(x, points[:, k])]) , [0])
#
#        kern_discr = kern_app_point(mg)
#
#        unstructured += space.tangent_bundle.element([kern_discr * vect for vect in vectors[:, k]]).copy()
#
#    return unstructured


get_unstructured_op_generate = struct.get_from_structured_to_unstructured(space, kernel_generate)

def calibration_init(original, noisy, group, action, product, pairing):
    """
    Main calibration function.
    """
    norm_original = get_unstructured_op_generate(original).norm()
    diff = np.abs(noisy) - np.abs(get_unstructured_op_generate(original))
    ide = space.element(space.points())
    return proj(ide.inner(space.element(diff)) / norm_original)

def calibration(original, noisy, group, action, product, pairing):
    result = cali.calibrate(original, noisy, group, action, product, pairing)
    return result.x


#%% define data

dim = 1
points_truth = np.array([[-0.6, 0.0, 0.2, 0.5,]])
vectors_truth = np.array([[0.3, 0.0, 0, 1,]])
original = struct.create_structured(points_truth, vectors_truth)
original_unstructured = get_unstructured_op_generate(original)
data_list = []
nb_data = 10
translation_list = np.random.uniform(low=-1.0, high=1.0, size = nb_data)
covariance_matrix = struct.make_covariance_matrix(space.points().T, kernel_generate)
noise_l2 =  odl.phantom.noise.white_noise(odl.ProductSpace(space, nb_data))*0.05
decomp = np.linalg.cholesky(covariance_matrix + 1e-4 * np.identity(len(covariance_matrix)))
noise_rkhs = [np.dot(decomp, noise_l2[i]) for i in range(nb_data)]
pts_space=space.points().T
#data_list=[]
#for i in range(nb_data):
#    pts_displaced = g.apply(np.array([-translation_list[i]]), pts_space)
#    data_list.append(space.tangent_bundle.element([original_unstructured[u].interpolation(pts_displaced) for u in range(dim)]))



data_list_noisy = [space.tangent_bundle.element(get_unstructured_op_generate(action(np.array([translation_list[i]]), original)) + noise_rkhs[i]) for i in range(nb_data)]
data_list = [space.tangent_bundle.element(get_unstructured_op_generate(action(np.array([translation_list[i]]), original))) for i in range(nb_data)]

#
#for i in range(nb_data):
#    (data_list[i] -data_list_bis[i]).show()
##    space.element(noise_rkhs[i]).show()
###
#for i in range(nb_data):
#    original_unstructured.show('original')
#    print(translation_list[i])
#    data_list[i].show(str(i))
#    data_list_bis[i].show('bis' + str(i))
##    space.element(noise_rkhs[i]).show()
###
##
#for i in range(nb_data):
#    data_list_bis[i].show('bis' + str(i))
##    space.element(noise_rkhs[i]).show()
##
#%%
dim = 1
nb_iteration = 20
points = np.array(points_list).T
# first raw estimation
result = scheme.iterative_scheme(solve_regression, calibration, action, g,
                                 kernel, data_list, sigma0,
                                 sigma1, points, nb_iteration)
#pts_space=space.points().T
#data_displaced = []
#for i in range(nb_data):
#    # Apply inverse : specific here !
#    pts_displaced = g.apply(result_init[1][i], pts_space)
#    data_displaced.append(space.tangent_bundle.element([data_list[i][u].interpolation(pts_displaced) for u in range(dim)]))
#
#for i in range(nb_data):
#    data_displaced[i].show()
#
#%%

original_computed_unstructured = get_unstructured_op(result[0])
original_computed_unstructured.show()
original_unstructured.show()
#%%
#np.savetxt('/home/bgris/DeformationModulesODL/deform/vect_field_rotation_SheppLogan_scheme_sigma_0_3__nbtrans_72',vect_field_ref)


#
##%%
#eval_kernel = struct.make_covariance_matrix(points, kernel)
#
#group_element_init = g.identity
#vectors_original = solve_regression(g, [group_element_init], [data_list[0]], sigma0, sigma1, points, eval_kernel)
#vectors_original_struct = struct.get_structured_vectors_from_concatenated(vectors_original, nbtrans, dim)
#original = struct.create_structured(points, vectors_original_struct)
#
#velo = calibration(original, data_list[1], g, action, product, pairing)
#print(velo)
#depl = action(g.exponential(velo), original)
#depl_computed_unstructured = get_unstructured_op_generate(depl)
#depl_computed_unstructured.show('computed')
#data_list[1].show('data')
#
#original_computed_unstructured = get_unstructured_op(original)
#original_computed_unstructured.show()
