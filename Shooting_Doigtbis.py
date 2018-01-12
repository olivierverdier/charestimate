#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:16:47 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:17:38 2018

@author: bgris
"""



import sys
sys.path.insert(0, '/home/bgris')
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
import function_generate_data_doigt_bis as fun_gen

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


r_b = 4
r_c = 4
sigma = 0.2

nbdata = 10
pathresult = '/home/bgris/Results/DeformationModules/Doigtbis/'
#pathresult = '/home/bgris/Results/DeformationModules/RotationRectangle/'
#pathdata = '/home/bgris/data/RotationRectangle/'
pathdata = '/home/bgris/data/Doigtbis/'
name_exp = 'rb_' + str(r_b) +'_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) +  '_nbdata_' + str(nbdata)


namedata = pathdata + name_exp
nameresult = pathresult + name_exp


nb_ab = int(np.loadtxt(namedata + '/nameab'))
nb_bc = int(np.loadtxt(namedata + '/namebc'))
n_orth = int(np.loadtxt(namedata + '/name_orth'))
alpha = np.loadtxt(nameresult + 'alpha')

def kernel(x, y):
    #si = tf.shape(x)[0]
    return np.exp(- sum([ (x[i] - y[i]) ** 2 for i in range(dim)]) / (sigma ** 2))




def generate_points_vectors(o):
    a = o[0].copy()
    b = o[1].copy()
    c = o[2].copy()
    return cmp.compute_pointsvectors_2articulations_bis_nb(a, b, c, width, sigma, n_orth, nb_ab, nb_bc)


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


#

#%%


nb_basepoints = 3
Articul2 = FromPointsVectorsCoeff.FromPointsVectorsCoeff(space, nb_basepoints, generate_points_vectors,
                                                          compute_vectorfield_pointsvectorcoeff, alpha)



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
theta_c_init = 0.1*np.pi

a_init =[0., 0.]
b_init = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c*np.cos(theta_b_init + theta_c_init), b_init[1] + r_c*np.sin(theta_b_init + theta_c_init)]

a = [0., 0.]
b = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c = [b[0] + r_c*np.cos(theta_b_init+theta_c_init), b[1] + r_c*np.sin(theta_b_init+theta_c_init)]



template_init = fun_gen.generate_image_2articulations(space, a, b, c, width)
template = space.element(scipy.ndimage.filters.gaussian_filter(template_init.asarray(),1))
template.show()
plt.plot(a[0], a[1],'xr')
plt.plot(b[0], b[1],'xr')
plt.plot(c[0], c[1],'xr')


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
Cont = 0.5*odl.ProductSpace(Module.Contspace, nb_time_point_int+1).one()

GD_mod, vect_field_mod=vect_field_list(GD_init, Cont)

#GD_exp = []
#GD_exp.append(GD_init)
#vect_field_exp = []
#
#for i in range(nb_time_point_int+1):
#    a = GD_exp[i][0]
#    b = GD_exp[i][1]
#    c = GD_exp[i][2]
#    vect_field_temp =  generate_vectorfield_2articulations_0(space, a, b, c, width).copy()
#    vect_field_exp.append(vect_field_temp.copy())
#    GD_speed = np.array([vect_field_temp[u].interpolation(GD_exp[i].T) for u in range(2)]).T
#    GD_exp.append(GD_exp[i] + (1/nb_time_point_int)*GD_speed)
#
#vect_field_exp = odl.ProductSpace(template.space.tangent_bundle,nb_time_point_int+1).element(vect_field_exp)
##

#%%
r_b_bis = 0.8*r_b
r_c_bis = 0.8*r_c
a_bis = [0., 0.]
b_bis = [a_init[0] + r_b_bis*np.cos(theta_b_init), a_init[1] + r_b_bis*np.sin(theta_b_init)]
c_bis = [b_bis[0] + r_c_bis*np.cos(theta_c_init+theta_b_init), b_bis[1] + r_c_bis*np.sin(theta_c_init+theta_b_init)]

template_bis = fun_gen.generate_image_2articulations(space, a_bis, b_bis, c_bis, 0.5*width)
template_bis = space.element(scipy.ndimage.filters.gaussian_filter(template_bis.asarray(),5))
template_bis.show()
I_mod=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_mod, template_bis)
#I_exp=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_exp, template)

for t in range(nb_time_point_int+1):
    I_mod[t].show('mod' + str(t), clim=[0,1])
    plt.plot(GD_mod[t].T[0], GD_mod[t].T[1],'xr')
#    I_exp[t].show('exp' + str(t), clim=[0.2, 1])
#    plt.plot(GD_exp[t].T[0], GD_exp[t].T[1],'xr')
#
plt.axis('equal')
