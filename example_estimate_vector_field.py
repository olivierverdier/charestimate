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

#%%
## Set parameters
# Size of dataset
size = 10


# noise of observation
sigmanoise=0.2

# scale of the kernel
sigma_kernel=0.3
fac=2
xmin=-2.2
xmax=3.2
dx=round((xmax-xmin)/(fac*sigma_kernel))
ymin=-2.0
ymax=2.0
dy=round((ymax-ymin)/(fac*sigma_kernel))
points=[]
for i in range(dx+1):
    for j in range(dy+1):
        points.append([xmin +fac*sigma_kernel* i*1.0, ymin + fac*sigma_kernel*j*1.0])
#        x0.append(xmin +fac*sigma_kernel* i*1.0)
#        x0.append(ymin + fac*sigma_kernel*j*1.0)

#Number of translations
nbtrans=round(0.5*len(points))

# to be of the good shape for iterative scheme
points = np.array(points).T

def kernel(x, y):
    return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x,y)]) / (sigma_kernel ** 2))

space=odl.uniform_discr(
min_pt=[-16, -16], max_pt=[16, 16], shape=[128,128],
dtype='float32', interp='linear')

vectorfield_list_center=[]
path='/home/bgris/data/SheppLoganRotationSmallDef/'
name='vectfield_smalldef_sigma_0_3'
for i in range(size):
    name_i=path + name + '_{}'.format(i)
    vect_field_load_i_test=space.tangent_bundle.element(np.loadtxt(name_i)).copy()
    vectorfield_list_center.append(vect_field_load_i_test.copy())

if False:
    vectorfield_list_center[0].show()
    plt.plot(points[0], points[1],'x')

#%%

g = group.ScaleDisplacement
solve_regression = reg.solve_regression
calibration = cali.calibrate
def action(group_element, structured_field):
    return act.apply_element_to_field(g, acc.create_signed_element(1, group_element), structured_field)

def product(vect0, vect1):
    return struct.scalar_product_structured(vect0, vect1, kernel)


pairing = struct.scalar_product_unstructured
sigma0 = 1
sigma1 = 10

nb_iteration = 10
field_list = copy.deepcopy(vectorfield_list_center)


result = scheme.iterative_scheme(solve_regression, calibration, action, g,
                                 kernel, vectorfield_list_center, sigma0,
                                 sigma1, points, nb_iteration)




