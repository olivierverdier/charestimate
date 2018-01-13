#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:47:04 2018

@author: barbara
"""

#namepath = 'bgris'
namepath = 'barbara'

import sys
sys.path.insert(0, '/home/' + namepath)
import odl
import numpy as np

##%% Create data from lddmm registration
import matplotlib.pyplot as plt


from DeformationModulesODL.deform import Kernel
from DeformationModulesODL.deform import DeformationModuleAbstract
from DeformationModulesODL.deform import SumTranslations
from DeformationModulesODL.deform import UnconstrainedAffine
from DeformationModulesODL.deform import LocalScaling
from DeformationModulesODL.deform import LocalRotation
from DeformationModulesODL.deform import EllipseMvt
from DeformationModulesODL.deform import FromFile
from DeformationModulesODL.deform import FromFileV5
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom
from DeformationModulesODL.deform import FromPointsVectorsCoeff_MultiDim
import odl
import scipy.ndimage as ndimage

import numpy as np
from DeformationModulesODL.deform import FromPointsVectorsCoeff

import charestimate as char
import estimate_structured_base_pointsvectors as est_coeff
import function_compute_pointsvectors as cmp
#import generate_data_doigt as gen
import structured_vector_fields as struct
import function_generate_data_doigt_bis as fun_gen

import scipy


# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-10, -10], max_pt=[10, 10], shape=[512,512],
    dtype='float32', interp='linear')


width = 2

r_b = 4
r_c = 4
sigma = 0.2

nbdata = 10
mg = space.meshgrid
dim = 2

#%% Figure data
def kernel_np(x, y):
    #si = tf.shape(x)[0]
    return np.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(dim)]) / (sigma ** 2))


sblur = 5
namepath = 'bgris'
namepath = 'barbara'
path = '/home/' + namepath + '/data/'
pathresult = '/home/' + namepath + '/Results/DeformationModules/'
pathexp = 'Doigtbis_dimcont2/'
#pathexp = 'RotationTranslationRectangle_dimcont2/'

path += pathexp
#path = '/home/bgris/data/Doigtbis/'
#name_exp = 'rb_' + str(r_b) +  '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata) + '_sblur_' + str(sblur)
#name_exp = 'rb_' + str(r_b) + '_width_' + str(width) + '_sigma_' + str(sigma) + 'nb_fixed' + '_nbdata_' + str(nbdata)
#name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)

name = path + name_exp + '/'

structured_list = []
unstructured_list = []
points_list = []
vectors_list = []
cov_mat_list = []
param_list = []
A_inner_prod_list = []
image_list = []
nbdatamax = 5
for i in range(nbdatamax):
    structured_list.append(np.loadtxt(name + 'structured' + str(i)))
    unstructured_list.append(np.loadtxt(name + 'unstructured' + str(i)))
    vectors_i = structured_list[i][dim:2*dim]
    points_list.append(np.loadtxt(name + 'points' + str(i)))
    vectors_list.append(np.loadtxt(name + 'vectors' + str(i)))
    param_tmp = np.loadtxt(name + 'param' + str(i))
    param_list.append(param_tmp)
    image_temp = fun_gen.generate_image_2articulations(space, param_tmp[0:2], param_tmp[2:4],param_tmp[2:4] + 0.85*( param_tmp[4:6] - param_tmp[2:4]), 0.5*width).copy()
    image_list.append(space.element(scipy.ndimage.filters.gaussian_filter(image_temp, 10)))
    cov_mat_list.append(struct.make_covariance_matrix(points_list[i], kernel_np))
    A_inner_prod_list.append(np.dot(cov_mat_list[i], np.dot(vectors_i.T, vectors_list[i] )).T)
#

param_list = np.array(param_list).T

nb_points = len(points_list[0][0])
nb_vectors = len(vectors_list[0][0])


points = space.points()
step = 40
fac= 0.1
import os
namefig_init = pathresult + pathexp + name_exp


#%% Data
#gen_unstructured = struct.get_from_structured_to_unstructured(space, kernel_np)


#os.mkdir(namefig_init)

# save images
for i in range(nbdatamax):
    namefig = namefig_init + '/' + 'data_image_' + str(i)
    fig = image_list[i].show(clim=[0,1])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(namefig)
    
#

# save images + vector fields
for i in range(nbdatamax):
    namefig = namefig_init + '/' + 'data_image_vectfield' + str(i)
    fig = image_list[i].show(clim=[0,1])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    v = unstructured_list[i].copy()
    #v = gen_unstructured(structured_list[i])
    plt.quiver(points.T[0][::step],points.T[1][::step],fac*v[0][::step],fac*v[1][::step], color='r')

    plt.savefig(namefig)
    
#

# save images + GD
for i in range(nbdatamax):
    namefig = namefig_init + '/' + 'data_image_GD' + str(i)
    fig = image_list[i].show(clim=[0,1])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.plot(param_list.T[i][::2], param_list.T[i][1::2],'xb')
    plt.savefig(namefig)
    
#

# save images + vector fields + GD
for i in range(nbdatamax):
    namefig = namefig_init + '/' + 'data_image_vectfield_GD' + str(i)
    fig = image_list[i].show(clim=[0,1])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    v = unstructured_list[i].copy()
    #v = gen_unstructured(structured_list[i])
    plt.quiver(points.T[0][::step],points.T[1][::step],fac*v[0][::step],fac*v[1][::step], color='r')

    plt.plot(param_list.T[i][::2], param_list.T[i][1::2],'xb')
    plt.savefig(namefig)
    
#

#%% Rsource and target

image = space.element(np.loadtxt('/home/barbara/data/Doigtbis_dimcont2/ex_source/11'))
image = space.element(scipy.ndimage.filters.gaussian_filter(image, 10))
namefig = '/home/barbara/data/Doigtbis_dimcont2/ex_source/11.png'
fig = image.show(clim=[0,1])
plt.axis('off')
fig.delaxes(fig.axes[1])
plt.savefig(namefig)





    