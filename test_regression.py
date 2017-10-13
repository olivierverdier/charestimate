#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:29:37 2017

@author: bgris
"""

import odl
import regression as reg
import numpy as np
import group
import structured_vector_fields as struct


def test_solve_regression():
    sigma = 1.0
    
    def kernel(x,y):
        return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(x,y)]) / (sigma ** 2))
        
    def kernel_app_0(x):
        return kernel(x, [0,0])
       
    scale = 0
    tx = 0.0
    ty = 0.0
    angle=np.pi/2
    infinitesimal0 = (scale, (angle, tx, ty))
    infinitesimal1 = (scale, (angle, tx, ty))
    signed_group_element0=(1, group.exponential(infinitesimal0))
    signed_group_element1=(-1, group.exponential(infinitesimal1))
    space = odl.uniform_discr(
            min_pt =[-10,-10], max_pt=[10,10], shape=[128, 128],
            dtype='float32', interp='linear')

    mg = space.meshgrid
    kern = kernel_app_0([mgu for mgu in mg])
    alpha_ref=[1.0, 0.0]
    #vector_field_ref = space.tangent_bundle.element([kern * hu for hu in alpha_ref]).copy()
    
    alpha0=[0.0, -1.0]
    vector_field0 = space.tangent_bundle.element([kern * hu for hu in alpha0]).copy()
    
    alpha1=[0.0, 1.0]
    vector_field1= space.tangent_bundle.element([kern * hu for hu in alpha1]).copy()
    
    sigma0 = 1.0
    sigma1 = 100.0
    
    points_a = np.array([[0.0],[0.0]])
    np_pts_a=len(points_a.T)
    points_b = np.array([[-1.0, 0.0, 1.0],[0.0, 0.0, 0.0]])
    np_pts_b=len(points_b.T)
    
    signed_group_element_list= [signed_group_element0, signed_group_element1]
    field_list=[vector_field0, vector_field1]
    
    eval_kernel_a=[[kernel(points_a[:,i], points_a[:,j]) for i in range(np_pts_a)] for j in range(np_pts_a)]
    eval_kernel_b=[[kernel(points_b[:,i], points_b[:,j]) for i in range(np_pts_b)] for j in range(np_pts_b)]

    alpha_est0 = reg.solve_regression(signed_group_element_list, field_list, sigma0, sigma1, points_a, eval_kernel_a)
    alpha_est1 = reg.solve_regression(signed_group_element_list, field_list, sigma0, sigma1, points_b, eval_kernel_b)
    
    print('alpha_est0 ={} '.format(alpha_est0))
    print('alpha_est1 = {}'.format(alpha_est1))
    
    
    
    
    
