#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:21:25 2017

@author: bgris
"""
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

a = [0, 0]
b = [0, 2]
c = [-2, 5]
mini = -5
maxi = 10

width = 1
sigma = 0.1

points, vectors = cmp.compute_pointsvectors_2articulations(a, b, c, width, sigma)
vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(a, b)
vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(b, c)

nb_ab = int((ab_norm + 0.2*width) / sigma) +1
nb_ab_orth = int(2 * width / sigma) +1
nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
nb_bc_orth = int(2*width / sigma) +1

pointsbis, vectorsbis = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)


plt.figure()
plt.plot(points[0], points[1], 'xb')
plt.plot(pointsbis[0], pointsbis[1], 'xg')
plt.plot(a[0], a[1], 'or')
plt.plot(b[0], b[1], 'or')
plt.plot(c[0], c[1], 'or')
plt.axis([mini, maxi, mini, maxi]), plt.grid(True, linestyle='--')



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
vector_ab_unit, vector_ab_norm_orth, ab_norm = gen.compute_vect_unit(a_list[0], b_list[0])
vector_bc_unit, vector_bc_norm_orth, bc_norm = gen.compute_vect_unit(b_list[0], c_list[0])
nb_ab = int((ab_norm + 0.2*width) / sigma) +1
nb_ab_orth = int(2 * width / sigma) +1
nb_bc = int((bc_norm  + 0.2*width) / sigma) +1
nb_bc_orth = int(2*width / sigma) +1

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
nbdatamax = 1

if False:
    for i in range(nbdatamax):
        a = a_list[i]
        b = b_list[i]
        c = c_list[i]
        points, vectors = cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)
        vector_translations = np.array([sum([alpha[u::nb_vectors]*vectors[v, u] for u in range(nb_vectors)]) for v in range(dim)])
        structured_i = struct.create_structured(points, vector_translations)
        unstructured_i = gen_unstructured(structured_i)
        (unstructured_i - vector_fields_list[i]).show('difference '+str(i))
        #(unstructured_i).show('computed '+str(i))
        #( vector_fields_list[i]).show('true '+str(i))
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



#%%


import sys
sys.path.insert(0, '/home/barbara')
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
import odl

import matplotlib.pyplot as plt
import numpy as np
from DeformationModulesODL.deform import FromPointsVectorsCoeff

import charestimate as char
import estimate_structured_base_pointsvectors as est_coeff
import function_compute_pointsvectors as cmp
#import generate_data_doigt as gen
import structured_vector_fields as struct

space = odl.uniform_discr(
        min_pt =[-10, -10], max_pt=[10, 10], shape=[128, 128],
        dtype='float32', interp='linear')


width = 1

#a = [0, 0]
#theta_b = 0.5*np.pi
#theta_c = 0.5*np.pi
#r_b = 2
#r_c = 5
#a, b, c = generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c)
#gen.generate_vectorfield_2articulations_0(space, a, b, c, width).show()


r_b = 4
r_c = 4
sigma = 0.5

nbdata = 10

pathresult = '/home/barbara/Results/DeformationModules/Doigt/'
pathdata = '/home/barbara/data/Doigt/'
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
namedata = pathdata + name_exp
nameresult = pathresult + name_exp

nb_ab = int(np.loadtxt(namedata + '/nameab'))
nb_ab_orth = int(np.loadtxt(namedata + '/nameaborth'))
nb_bc = int(np.loadtxt(namedata + '/namebc'))
nb_bc_orth = int(np.loadtxt(namedata + '/namebc_orth'))

alpha = np.loadtxt(nameresult + 'alpha')

def kernel(x, y):
    #si = tf.shape(x)[0]
    return np.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(dim)]) / (sigma ** 2))




def generate_points_vectors(o):
    a = o[0].copy()
    b = o[1].copy()
    c = o[2].copy()
    return cmp.compute_pointsvectors_2articulations_nb(a, b, c, width, sigma, nb_ab, nb_ab_orth, nb_bc, nb_bc_orth)


gen_unstructured = struct.get_from_structured_to_unstructured(space, kernel)

# if alpha is a vector
#def compute_vectorfield_pointsvectorcoeff(points, vectors, alpha):
#    nb_vectors = vectors.shape[1]
#    vector_translations = np.array([sum([alpha[u::nb_vectors]*vectors[v, u] for u in range(nb_vectors)]) for v in range(dim)])
#    structured = struct.create_structured(points, vector_translations)
#    unstructured = gen_unstructured(structured)
#
#    return unstructured.copy()

# if alpha is a matrix
def compute_vectorfield_pointsvectorcoeff(points, vectors, alpha):
    #nb_vectors = vectors.shape[1]
    vector_translations = np.dot(np.array(vectors), np.array(alpha))
    structured = struct.create_structured(points, vector_translations)
    unstructured = gen_unstructured(structured)

    return unstructured.copy()
#
#%%


nb_basepoints = 3
Articul2 = FromPointsVectorsCoeff.FromPointsVectorsCoeff(space, nb_basepoints, generate_points_vectors,
                                                          compute_vectorfield_pointsvectorcoeff, alpha)

#%%

#Articul2.ComputeField([a_test, b_test, c_test], [1]).show('artil')
#%%

#gen.generate_vectorfield_2articulations_0(space, a_test, b_test, c_test, width).show('gen')



#%% functional
dim = 2
lam=0.0001
nb_time_point_int=20
lamb0=1e-5
lamb1=1e-5

Module = Articul2
forward_op=odl.IdentityOperator(space)

import scipy
i=0
#a = param_list.T[i][0:2]
#b = param_list.T[i][2:4]
#c = param_list.T[i][4:6]


theta_b_init = 0.5*np.pi
theta_c_init = 0.25*np.pi

a_init =[0., 0.]
b_init = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c*np.cos(theta_c_init + theta_b_init), b_init[1] + r_c*np.sin(theta_c_init + theta_b_init)]

a = [0., 0.]
b = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c = [b_init[0] + r_c*np.cos(theta_c_init + theta_b_init), b_init[1] + r_c*np.sin(theta_c_init + theta_b_init)]



template_init = generate_image_2articulations(space, a, b, c, width)
template = space.element(scipy.ndimage.filters.gaussian_filter(template_init.asarray(),1))
template.show()
proj_data = forward_op(template)

# Add white Gaussion noise onto the noiseless data
noise =0.25 * odl.phantom.noise.white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

data = [noise_proj_data]
data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element(data)
forward_operators=[forward_op]

Norm=odl.solvers.L2NormSquared(forward_op.range)

functional_mod_temp = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lamb0, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)

#nb_ab


def vect_field_list(GD_init,Cont):
    space_pts=template.space.points()
    GD=GD_init.copy()
    vect_field_list_tot=[]
    GD_list = []
    GD_list.append(GD_init.copy())
    for i in range(nb_time_point_int+1):
        vect_field_mod=functional_mod_temp.Module.ComputeField(GD,Cont[i]).copy()
        vect_field_list_interp=template.space.tangent_bundle.element([vect_field_mod[u].interpolation(space_pts.T) for u in range(dim)]).copy()
        GD+=(1/nb_time_point_int)*functional_mod_temp.Module.ApplyVectorField(GD,vect_field_list_interp).copy()
        vect_field_list_tot.append(vect_field_list_interp)
        GD_list.append(GD.copy())

    return [GD_list, odl.ProductSpace(template.space.tangent_bundle,nb_time_point_int+1).element(vect_field_list_tot)]
#

#%%
GD_init=np.array([a_init, b_init, c_init])
Cont = 1*odl.ProductSpace(Module.Contspace, nb_time_point_int+1).one()

GD, vect_field=vect_field_list(GD_init, Cont)

I=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field, template)

for t in range(nb_time_point_int+1):
    I[t].show(str(t))
    plt.plot(GD[t].T[0], GD[t].T[1],'xr')
#
plt.axis('equal')

#%%
r_cbis = 0.8 * r_c
a_initbis =[-0.5, 0.]
b_initbis = [a_initbis[0] + r_b*np.cos(theta_b_init), a_initbis[1] + r_b*np.sin(theta_b_init)]
c_initbis = [b_initbis[0] + r_cbis*np.cos(theta_c_init + theta_b_init), b_initbis[1] + r_cbis*np.sin(theta_c_init + theta_b_init)]

templatebis = generate_image_2articulations(space, a_initbis, b_initbis, c_initbis, 0.5*width)
I=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field, templatebis)

for t in range(nb_time_point_int+1):
    I[t].show(str(t))
    plt.plot(GD[t].T[0], GD[t].T[1],'xr')
#
plt.axis('equal')
#%%

for i in range(nb_time_point_int):
    vect_field[i].show(str(i))
#%% Faire figures vector fields data
points = space.points()
path = '/home/barbara/data/Doigt/'
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata) + '/'
name = path + name_exp
name_exp_save = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_nbdata_' + str(nbdata)

get_unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)
nbdatamax = 5
step = 10
for idata in range(nbdatamax):
    namefig = pathresult + name_exp_save + 'data' + str(idata) + '.png'
    v = 0.5*get_unstructured_op(np.loadtxt(name + 'structured' + str(idata)))
    param = np.loadtxt(name + 'param' + str(idata))
    image = generate_image_2articulations(space, param[0:2], param[2:4], param[4:6], width)
    fig = image.show(clim=[0, 2])
    #    fig = plt.figure()
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.quiver(points.T[0][::step],points.T[1][::step],v[0][::step],v[1][::step], color='r', scale = 10)
    plt.plot(param[::2], param[1::2], 'xb')
    plt.savefig(namefig)
    
    
    namefig =  pathresult + name_exp_save + 'data' + str(idata) + 'estimated' + '.png'

    v =  0.25*Articul2.ComputeField([param[0:2], param[2:4], param[4:6]], [1])
    fig = image.show(clim=[0, 2])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.quiver(points.T[0][::step],points.T[1][::step],v[0][::step],v[1][::step], color='r', scale = 10)
    plt.plot(param[::2], param[1::2], 'xb')
    
    plt.savefig(namefig)
#

#%% faire figure  nouveaux parametres
namefig = pathresult + name_exp_save +  'estimatedtest1' + '.png'

theta_b_init = 0.2*np.pi
theta_c_init = 0.3*np.pi

a_init =[-5., 0.0]
b_init = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c*np.cos(theta_c_init + theta_b_init), b_init[1] + r_c*np.sin(theta_c_init + theta_b_init)]


v = 0.25*Articul2.ComputeField([a_init, b_init, c_init], [1])
image = generate_image_2articulations(space, a_init, b_init, c_init, width)
fig = image.show(clim=[0, 2])
#fig = plt.figure()
plt.axis('off')
fig.delaxes(fig.axes[1])
plt.quiver(points.T[0][::step],points.T[1][::step],v[0][::step],v[1][::step], color='r', scale = 10)
plt.plot(a_init[0], a_init[1], 'xb')
plt.plot(b_init[0], b_init[1], 'xb')
plt.plot(c_init[0], c_init[1], 'xb')

plt.savefig(namefig)

#%%
t=5
plt.figure()
points, vectors = generate_points_vectors(GD[t])
plt.plot(points[0], points[1], 'xb')
plt.plot(GD[t].T[0], GD[t].T[1], 'xr')


#%% Test trajectory with wanted vector fields
i0 = 1
a = np.array(a_list[i0].copy())
b = np.array(b_list[i0].copy())
c = np.array(c_list[i0].copy())

a=a_init.copy()
b=b_init.copy()
c=c_init.copy()

nb_time_point_int=10
vect_field_integration = []
delta_t = 1 / nb_time_point_int
for t in range(nb_time_point_int):
    #plt.figure(str(t))
    points, vectors = cmp.compute_pointsvectors_2articulations(a, b, c, width, sigma)
    #plt.plot(points[0], points[1], 'xb')

    plt.plot(a[0], a[1], 'or')
    plt.plot(b[0], b[1], 'or')
    plt.plot(c[0], c[1], 'or')
    #plt.axis([mini, maxi, mini, maxi]), plt.grid(True, linestyle='--')
    vect_field_temp = gen.generate_vectorfield_2articulations_0(space, a, b, c, width).copy()
    #vect_field_temp[0].show()

    vect_field_integration.append(vect_field_temp.copy())
    a = a + delta_t*np.array([v.interpolation(a) for v in vect_field_temp])
    b = b + delta_t*np.array([v.interpolation(b) for v in vect_field_temp])
    c = c + delta_t*np.array([v.interpolation(c) for v in vect_field_temp])

#
plt.axis('equal')

#%%

for t in range(nb_time_point_int):
    plt.figure(str(t))

    plt.plot(a[0], a[1], 'or')
    plt.plot(b[0], b[1], 'or')
    plt.plot(c[0], c[1], 'or')
    #plt.plot(points_list[0][0], points_list[0][1], 'xb')
    plt.axis([mini, maxi, mini, maxi]), plt.grid(True, linestyle='--')




