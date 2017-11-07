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
sigma_kernel=0.3
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
dtype='float32', interp='nearest')
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


#%% define data

dim = 1
nb_pt_generate = 3
points_truth = np.random.uniform(low=-1.0, high=1.0, size = nb_pt_generate)
vectors_truth = np.random.uniform(low=-1.0, high=1.0, size = nb_pt_generate)
original = struct.create_structured(points_truth, vectors_truth)
original_unstructured = get_unstructured_op_generate(original)
data_list = []
nb_data = 10
translation_list = np.random.uniform(low=-1.0, high=1.0, size = nb_data)
covariance_matrix = struct.make_covariance_matrix(space.points().T, kernel_generate)
noise_l2 =  odl.phantom.noise.white_noise(odl.ProductSpace(space, nb_data))*0.1
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
#
#for i in range(nb_data):
#    data_list_noisy[i].show('bis' + str(i))
##    space.element(noise_rkhs[i]).show()
##
#%%

#TODO : change the following for new calibration type
#def calibration_init(original, noisy, group, action, product, pairing):
#    """
#    Main calibration function.
#    """
#    norm_original = get_unstructured_op_generate(original).norm()
#    diff = np.abs(noisy) - np.abs(get_unstructured_op_generate(original))
#    ide = space.element(space.points())
#    return proj(ide.inner(space.element(diff)) / norm_original)
#
#def calibration(original, noisy, group, action, product, pairing):
#    result = cali.calibrate(original, noisy, group, action, product, pairing)
#    return result.x

def calibration_equation(original, noisy):
    result = cali.calibrate_equation_1D_translation(original, noisy, space, kernel)
    return result



sigma0 = 1
sigma1 = 500

dim = 1
nb_iteration = 30
points = np.array(points_list).T
# first raw estimation
result = scheme.iterative_scheme(solve_regression, calibration_equation, action, g,
                                 kernel, data_list_noisy, sigma0,
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


#%% Compare ground truth and result
result_unstruc = get_unstructured_op(result[0])
original_unstructured = get_unstructured_op_generate(original)
velo = calibration_equation(result[0], original_unstructured)
computed = action(g.exponential(velo), result[0])
result_unstruc_i = get_unstructured_op(computed)
plt.figure()
plt.plot(space.points().T[0], original_unstructured[0].asarray(), label = 'ground truth')
plt.plot(space.points().T[0], result_unstruc_i[0].asarray(), label = 'result calibrated')
plt.plot(space.points().T[0], result_unstruc[0].asarray(), label = 'result ')
plt.legend()



#%% compare result and initialisation

original_computed_unstructured = get_unstructured_op(result[0])
#data_list_noisy[0][0].show('initialisation')
#original_computed_unstructured.show('result')
#original_unstructured.show('original')

plt.plot(space.points().T[0], data_list_noisy[0][0].asarray(), label = 'data noisy 0 ')
#plt.ylabel('init')
plt.axis([-1,1,0,1])


plt.plot(space.points().T[0], original_computed_unstructured[0].asarray(), label = 'computed regressed')
#plt.ylabel('result')
#plt.axis([-1,1,0,1])


plt.plot(space.points().T[0], original_unstructured[0].asarray(), label = 'original')
#plt.ylabel('original')
plt.axis([-1,1,0,1])
plt.legend()


#%% Compare result with all data
result_unstruc = get_unstructured_op(result[0])

for i in range(nb_data):
    velo = calibration_equation(result[0], data_list_noisy[i])
    computed = action(g.exponential(velo), result[0])
    result_unstruc_i = get_unstructured_op(computed)
    plt.figure()
    plt.plot(space.points().T[0], data_list_noisy[i][0].asarray(), label = 'data')
    plt.plot(space.points().T[0], result_unstruc_i[0].asarray(), label = 'result calibrated')
    plt.plot(space.points().T[0], result_unstruc[0].asarray(), label = 'result ')
    plt.legend()
#
#%%
#
#velo0 = calibration(original, original_computed_unstructured, g, action, product, pairing)
#velo1 = calibration_init(original, original_computed_unstructured, g, action, product, pairing)
#depl = action(g.exponential(velo), original)
#
#
#velo0 = calibration(original, data_list[1], g, action, product, pairing)
#velo1 = calibration_init(original, original_computed_unstructured, g, action, product, pairing)
#get_unstructured_op_generate(original).show()
#data_list[1].show()
#
#
#depl_computed_unstructured = get_unstructured_op_generate(depl)
#(depl_computed_unstructured - original_computed_unstructured).show()
#
#fig = (depl_computed_unstructured - original_computed_unstructured).show()
#depl_computed_unstructured.show()
#
#((depl_computed_unstructured - original_computed_unstructured)**2 / (original_computed_unstructured ** 2)).show()
##%%
#test0=get_unstructured_op(original)
#plt.plot(space.points().T[0], test0[0].asarray(), label = 'original regression')
#
#plt.plot(space.points().T[0],vect_field_list[0][0].asarray(), label = 'data')
#plt.axis([-1,1,0,1])
#plt.legend()
##%%
#ori_unstruc = get_unstructured_op(original)
#for i in range(nb_data):
#    computed = action(group_element_list[i],original)
#    result_unstruc_i = get_unstructured_op(computed)
#    plt.figure()
#    plt.plot(space.points().T[0], data_list_noisy[i][0].asarray(), label = 'data')
#    #plt.plot(space.points().T[0], result_unstruc_i[0].asarray(), label = 'result calibrated')
#    plt.plot(space.points().T[0], ori_unstruc[0].asarray(), label = 'ori ')
#    plt.legend()
#
#%%
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
