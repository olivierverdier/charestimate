#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:12:10 2018

@author: bgris
"""

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
from DeformationModulesODL.deform import FromPointsVectorsCoeff_MultiDim
import odl
import scipy.ndimage as ndimage
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


width = 2

#a = [0, 0]
#theta_b = 0.5*np.pi
#theta_c = 0.5*np.pi
#r_b = 2
#r_c = 5
#a, b, c = generate_GD_from_athetas(a, theta_b, theta_c, r_b, r_c)
#gen.generate_vectorfield_2articulations_0(space, a, b, c, width).show()


r_b = 4
r_c = 4
sigmadata = 0.2
sigma = 0.5

nbdata = 10
nbdatamax = 5

sblur = 5
path = '/home/bgris/data/'

pathexp = 'Doigtbis_dimcont2/'
#pathexp = 'RotationTranslationRectangle_dimcont2/'

path += pathexp
#path = '/home/bgris/data/Doigtbis/'
#name_exp = 'rb_' + str(r_b) +  '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigmadata) + '_nbdata_' + str(nbdata) + '_sblur_' + str(sblur)


pathresult = '/home/bgris/Results/DeformationModules/Doigtbis_dimcont2/'
#pathresult = '/home/bgris/Results/DeformationModules/RotationRectangle/'
#pathdata = '/home/bgris/data/RotationRectangle/'
pathdata = '/home/bgris/data/Doigtbis_dimcont2/'
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigmadata) + '_nbdata_' + str(nbdata) + '_sblur_' + str(sblur)


namedata = pathdata + name_exp
nameresult = pathresult + name_exp


nb_ab = int(np.loadtxt(namedata + '/nameab'))
nb_bc = int(np.loadtxt(namedata + '/namebc'))
n_orth = int(np.loadtxt(namedata + '/name_orth'))
alpha = np.array([np.loadtxt(nameresult + 'alpha' + str(u)) for u in range(2)])

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
Articul2 = FromPointsVectorsCoeff_MultiDim.FromPointsVectorsCoeff_MultiDim(space, nb_basepoints, generate_points_vectors,
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
theta_c_init = 0.3*np.pi

a_init =[0., 0.]
b_init = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c*np.cos(theta_b_init + theta_c_init), b_init[1] + r_c*np.sin(theta_b_init + theta_c_init)]

a = [0., 0.]
b = [a_init[0] + r_b*np.cos(theta_b_init), a_init[1] + r_b*np.sin(theta_b_init)]
c = [b[0] + r_c*np.cos(theta_b_init+theta_c_init), b[1] + r_c*np.sin(theta_b_init+theta_c_init)]



template_init = fun_gen.generate_image_2articulations(space, a, b, c, 0.1*width)
template = template_init.copy()
#template = 10*space.element(scipy.ndimage.filters.gaussian_filter(template_init.asarray(), 10))
template.show('template', clim = [0,1])
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
theta_b_init = 0.5*np.pi
theta_c_init = 0.0*np.pi

r_b_bis = r_b
r_c_bis = 1.2*r_c
a_init =[0., -2.]
b_init = [a_init[0] + r_b_bis*np.cos(theta_b_init), a_init[1] + r_b_bis*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c_bis*np.cos(theta_b_init + theta_c_init), b_init[1] + r_c_bis*np.sin(theta_b_init + theta_c_init)]

GD_init=np.array([a_init, b_init, c_init])
Cont = 0.2*odl.ProductSpace(Module.Contspace, nb_time_point_int+1).one()

for i in range(nb_time_point_int):
    Cont[i][0]*= 1.1
    Cont[i][1]*= 1.1

GD_mod, vect_field_mod = vect_field_list(GD_init, Cont)

#%%
#width= 2
#
#Cont0 = 0.5
#Cont1 = 1
#GD_exp = []
#GD_exp.append(GD_init)
#vect_field_exp = []
##
#sblur = 5
#for i in range(nb_time_point_int+1):
#    a = GD_exp[i][0]
#    b = GD_exp[i][1]
#    c = GD_exp[i][2]
#    vect_field_temp0 =  fun_gen.generate_vectorfield_2articulations_0(space, a, b, c, width).copy()
#    vect_field_temp0 =  [ndimage.gaussian_filter(vect_field_temp0[u], sigma = (sblur,sblur),  order=0) for u in range(dim)]
#    vect_field_temp0 = space.tangent_bundle.element(vect_field_temp0)
#    vect_field_temp1 =  fun_gen.generate_vectorfield_2articulations_1(space, a, b, c, width).copy()
#    vect_field_temp1 =  [ndimage.gaussian_filter(vect_field_temp1[u], sigma = (sblur,sblur),  order=0) for u in range(dim)]
#    vect_field_temp1 = space.tangent_bundle.element(vect_field_temp1)
#    vect_field_exp.append(Cont0 * vect_field_temp0 + Cont1 * vect_field_temp1)
#    GD_speed = np.array([vect_field_exp[i][u].interpolation(GD_exp[i].T) for u in range(2)]).T
#    GD_exp.append(GD_exp[i] + (1/nb_time_point_int)*GD_speed)
#
#vect_field_exp = odl.ProductSpace(template.space.tangent_bundle,nb_time_point_int+1).element(vect_field_exp)
##

##%%
#r_b_bis = 0.8*r_b
#r_c_bis = 0.8*r_c
#a_bis = [0., 0.]
#b_bis = [a_init[0] + r_b_bis*np.cos(theta_b_init), a_init[1] + r_b_bis*np.sin(theta_b_init)]
#c_bis = [b_bis[0] + r_c_bis*np.cos(theta_c_init+theta_b_init), b_bis[1] + r_c_bis*np.sin(theta_c_init+theta_b_init)]
#
#template_bis = fun_gen.generate_image_2articulations(space, a_bis, b_bis, c_bis, 0.5*width)
#template_bis = space.element(scipy.ndimage.filters.gaussian_filter(template_bis.asarray(),5))
#template_bis.show()
#I_mod=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_mod, template_bis)
##%%
r_b_bis = r_b
r_c_bis = 1.1*r_c
a_init =[0., -2.]
b_init = [a_init[0] + r_b_bis*np.cos(theta_b_init), a_init[1] + r_b_bis*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c_bis*np.cos(theta_b_init + theta_c_init), b_init[1] + r_c_bis*np.sin(theta_b_init + theta_c_init)]

template_init = fun_gen.generate_image_2articulations(space, a_init, b_init, c_init, 1*width)
template = template_init.copy()
template = 1*space.element(scipy.ndimage.filters.gaussian_filter(template_init.asarray(), 10))
I_mod=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_mod, template)

#I_exp=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_exp, template)
##%%
for t in range(nb_time_point_int+1):
    I_mod[t].show('mod' + str(t), clim=[0,1])
    plt.plot(GD_mod[t].T[0], GD_mod[t].T[1],'xr')
    #I_exp[t].show('exp' + str(t), clim=[0.1, 1])
    #plt.plot(GD_exp[t].T[0], GD_exp[t].T[1],'xr')
    plt.axis([-10.0, 10.0, -10.0, 10.0])
    plt.axis('equal')
#
#plt.axis('equal')

#plt.plot(a[0], a[1], 'x')
#plt.plot(b[0], b[1], 'x')
#plt.plot(c[0], c[1], 'x')

#grid_points=compute_grid_deformation_list(vect_field_mod, 1/nb_time_point_int, template.space.points().T)

#%%
import copy
I_mod_save = copy.deepcopy(I_mod)
#%%
for t in range(nb_time_point_int+1):
    I_mod[t].show('mod' + str(t), clim=[0,1])
    plt.plot(GD_mod[t].T[0], GD_mod[t].T[1],'xr')
    #I_exp[t].show('exp' + str(t), clim=[0.1, 1])
    #plt.plot(GD_exp[t].T[0], GD_exp[t].T[1],'xr')
    #grid = grid_points[t].reshape((2,template.space.shape[0], template.space.shape[1]))
    #plot_grid(grid, 5)
    #plt.axis([-10.0, 10.0, -10.0, 10.0])
    plt.axis('equal')

#
#%%
i=5
a = GD_exp[i][0]
b = GD_exp[i][1]
c = GD_exp[i][2]
fun_gen.generate_image_2articulations_vectfield_1(space, a, b, c, width).show( str(i))
plt.plot(GD_exp[i].T[0], GD_exp[i].T[1],'xr')
vect_field_exp[i].show(str(i))

#%%

Cont_init = Articul2.Contspace.element([1,-1])
vect_field_mod=functional_mod_temp.Module.ComputeField(GD_init,Cont_init).copy()
vect_field_mod.show(str(Cont_init))

#%%
r_b_bis = 1.5*r_b
r_c_bis = 1.5*0.9*r_c
a_init =[0., 0.]
b_init = [a_init[0] + r_b_bis*np.cos(theta_b_init), a_init[1] + r_b_bis*np.sin(theta_b_init)]
c_init = [b_init[0] + r_c_bis*np.cos(theta_b_init + theta_c_init), b_init[1] + r_c_bis*np.sin(theta_b_init + theta_c_init)]
GD_init=np.array([a_init, b_init, c_init])



Cont_init = Articul2.Contspace.element([1,0])
v10 = Articul2.ComputeField(GD_init, Cont_init)

v10.show()


#%%

Cont = [1., 1.]
v = Cont[0] * v10 + Cont[1] * v01


points=space.points()
fig = template.show(str(Cont), clim=[0, 2])
plt.axis('off')
fig.delaxes(fig.axes[1])

step = 60
fac = 0.5
plt.quiver(points.T[0][::step],points.T[1][::step],fac*v[0][::step],fac*v[1][::step], color='r', scale = 80)

#

#%%

path = '/home/bgris/data/Doigtbis_dimcont2/ex_source/'

np.savetxt(path + '11', I_mod[0].asarray())
#%%
import os
sblur = 5
path = '/home/bgris/Results/DeformationModules/'

pathexp = 'Doigtbis_dimcont2/'
#pathexp = 'RotationTranslationRectangle_dimcont2/'

path += pathexp
#path = '/home/bgris/data/Doigtbis/'
#name_exp = 'rb_' + str(r_b) +  '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)
name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata) + '_sblur_' + str(sblur)
#name_exp = 'rb_' + str(r_b) + '_width_' + str(width) + '_sigma_' + str(sigma) + 'nb_fixed' + '_nbdata_' + str(nbdata)
#name_exp = 'rb_' + str(r_b) + '_rc_' + str(r_c) + '_width_' + str(width) + '_sigma_' + str(sigma) + '_nbdata_' + str(nbdata)

name = path + name_exp + 'increasedsize' +'/'
os.mkdir(name)

points=space.points()
step = 40
for t in range(nb_time_point_int+1):
    fig =I_mod[t].show( clim=[0,1])
    plt.plot(GD_mod[t].T[0], GD_mod[t].T[1],'xr')
    name_fig = name +  'plot_t_' + str(t)
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(name_fig)

#
#for k in range(NbMod):
#    plt.plot(GD[i][k][2][0], GD[i][k][2][1], 'xb')
#    plt.plot(GD[i][k][3][0], GD[i][k][3][1], 'xb')
#plt
