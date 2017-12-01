#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:23:12 2017

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


#%% Generate data

# Size of dataset
size = 10
space=odl.uniform_discr(
min_pt=[-16, -16], max_pt=[16, 16], shape=[512,512],
dtype='float32', interp='linear')
data_list=[]
path='/home/bgris/data/SheppLoganRotationSmallDef/vectfield512/'
name='vectfield_smalldef_sigma_0_3'
name_i=path + name + '_0'
original = space.tangent_bundle.element(np.loadtxt(name_i)).copy()

# rotations of the initial data


#%%

maxx=5.0
minx = -5
miny=-5
maxy=5

theta=np.pi/3
centre=np.array([0,0])

points=space.points()

def Rtheta(theta,points):
    # theta is the angle, in rad
    # input = list of points, for ex given by space.points() or
    # np.array(vect_field).T
    #output = list of points of same size, rotated of an angle theta

    points_rot=np.empty_like(points).T
    points_rot[0]=np.cos(theta)*points.T[0].copy() - np.sin(theta)*points.T[1].copy()
    points_rot[1]=np.sin(theta)*points.T[0].copy() + np.cos(theta)*points.T[1].copy()

    return points_rot.T.copy()
    points.T[0]

def Rot_vect_field_cache(minx,maxx,miny,maxy,theta,centre,vect_field):
    v1=space.tangent_bundle.element()
    for i in range(len(points)):
        #print(str(i))
        pt=points[i]
        if(pt[0]>minx and pt[0] < maxx and pt[1]>miny and pt[1]<maxy):

            pt_rot_inv=Rtheta(-theta,pt-centre).copy()
            valx = vect_field[0].interpolation([[pt_rot_inv[0]+centre[0]],[pt_rot_inv[1]+centre[1]]])
            valy = vect_field[1].interpolation([[pt_rot_inv[0]+centre[0]],[pt_rot_inv[1]+centre[1]]])

            v1[0][i] = np.cos(theta) * valx - np.sin(theta) * valy
            v1[1][i] = np.sin(theta) * valx + np.cos(theta) * valy

        else:
            v1[0][i]=vect_field[0][i]
            v1[1][i]=vect_field[1][i]

    return v1


v1=Rot_vect_field_cache(minx,maxx,miny,maxy,theta,centre,original)
v2 = Rot_vect_field_cache(minx,maxx,miny,maxy,-theta,centre,v1)
original.show('original')
v1.show('rotated')
(original - v2).show('difference')
#%%
theta_list=[0 , 15, -20, 30, -50, 45, -10, 20, -30]
theta_list=[0 , 10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
theta_list=np.pi*np.array([0 , 0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
nb_data=len(theta_list)
theta_dec=0.05*np.pi
images_source=[]
images_target=[]
for i in range(size):
    vect_field_rot
    images_source.append(Rot_image_cache(minx,maxx,miny,maxy,theta_list[i],centre,template).copy())
    images_target.append(Rot_image_cache(minx,maxx,miny,maxy,theta_list[i]+theta_dec,centre,template).copy())
#




data_list = []

for i in range(size):



#
#%% Set parameters kernel and control points

# scale of the kernel
sigma_kernel=0.3
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
nb_iteration = 5
points = np.array(points_list).T
# first raw estimation
result = scheme.iterative_scheme(solve_regression, calibration_equation, action, g,
                                 kernel, data_list, sigma0,
                                 sigma1, points, nb_iteration)
#
#%% Compare result with all data
result_unstruc = get_unstructured_op(result[0])
result_unstruc.show('computed {}'.format(i), clim = [-0.1, 0.1])

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