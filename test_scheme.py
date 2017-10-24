#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:41:58 2017

@author: bgris
"""


import numpy as np
import odl
import structured_vector_fields as struct
import calibration as calib
import group
import action
import regression as reg
import scheme

def projection_periodicity(space):
    maxi = space.max_pt[0]
    mini = space.min_pt[0]
    space_extent = maxi - mini

    def proj(point):
        return np.mod(point- space.min_pt, space_extent) + mini

    return proj

def get_kernel(space):
    proj = projection_periodicity(space)

    scale = 0.5*space.cell_sides
    #vol_cell = space.cell_volume
    def kernel(point0, point1):

        point0_periodic =proj(point0)
        point1_periodic = proj(point1)

        #value = 0.0
        mask = np.abs(point0_periodic - point1_periodic.T) < scale

        result = np.zeros_like(mask, dtype=float)
        result[mask] = 1.0

        return result

    return kernel



def get_kernel_gauss(space, sigma):
    proj = projection_periodicity(space)
    def kernel(point0, point1):

        point0_periodic = proj(point0)
        point1_periodic = proj(point1)
        return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(point0_periodic,point1_periodic)]) / (sigma ** 2))

    return kernel



space = odl.uniform_discr(
        min_pt =[-1], max_pt=[1], shape=[128],
        dtype='float32', interp='linear')

#kernel = get_kernel(space)
proj = projection_periodicity(space)
class Translation():
    def exponential(velocity):
        return velocity
    zero_velocity = np.array([0.0])
    identity = np.array([0.0])

    @classmethod
    def apply(self, group_element, points):
        return proj(points + group_element)

    @classmethod
    def apply_differential(self, group_element, vectors):
        return vectors.copy()

    @classmethod
    def apply_differential_transpose(self, group_element, vectors):
        return vectors.copy()


def get_action(space):
    proj = projection_periodicity(space)

    def act(translation, f):
        points = struct.get_points(f)
        vectors = struct.get_vectors(f)
        dim, nb_points = points.shape

        points_translated = np.array([[points[u][v] + translation[u] for v in range(nb_points)] for u in range(dim)])
        points_translated_projected = proj(points_translated)

        return struct.create_structured(points_translated_projected, vectors)

    return act





solve_regression = reg.solve_regression
calibration = calib.calibrate
g = Translation
kernel = get_kernel_gauss(space, 0.2)
sigma0 = 1
sigma1 = 100.0

mg = space.meshgrid

def kernel_app_0(x):
    return kernel(x, [0])
kern = kernel_app_0(mg)

alpha0=[1.0]
vector_field0 = space.tangent_bundle.element([kern * hu for hu in alpha0]).copy()

points1 = np.array([-0.3])
def kernel_app_1(x):
    return kernel(x, points1)
kern1 = kernel_app_1([mgu for mgu in mg])
alpha1=[1.0]
vector_field1= space.tangent_bundle.element([kern1 * hu for hu in alpha1]).copy()


points2 = np.array([0.5])
def kernel_app_2(x):
    return kernel(x, points2)

alpha2=[1.0]
kern2 = kernel_app_2([mgu for mgu in mg])
vector_field2= space.tangent_bundle.element([kern2 * hu for hu in alpha2]).copy()

sigma0 = 1.0
sigma1 = 100.0

field_list = [vector_field0, vector_field1, vector_field2]

points_a = np.array([[0.0]])
points_b = np.array([[-0.5, 0.0, 0.5]])

nb_iteration = 30


act = get_action(space)

def test_iterative_scheme():
    result_a = scheme.iterative_scheme(solve_regression, calibration, act, g, kernel,
                                      field_list, sigma0, sigma1, points_a, nb_iteration)


    result_b = scheme.iterative_scheme(solve_regression, calibration, act, g, kernel,
                                      field_list, sigma0, sigma1, points_b, nb_iteration)




#iterative_scheme(solve_regression, calibration, action, g, kernel, field_list, sigma0, sigma1, points, nb_iteration)
