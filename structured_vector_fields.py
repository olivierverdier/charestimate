#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:44:55 2017

@author: bgris
"""

import numpy as np

__all__ = ('generate_scalar_products', 'scalar_product_structured',
           'scalar_product_unstructured')


def get_homogeneous_points(structured_field):
    dim_double, nb_points = structured_field.shape
    dim = int(dim_double/2)
    homogeneous_points = np.empty([dim + 1, nb_points])
    homogeneous_points[0:dim] = structured_field[0:dim].copy()
    homogeneous_points[dim] = 1

    return homogeneous_points


def get_points(structured_field):
    dim_double, nb_points = structured_field.shape
    dim = int(dim_double/2)

    return structured_field[0:dim].copy()


def get_vectors(structured_field):
    dim_double, nb_points = structured_field.shape
    dim = int(dim_double/2)

    return structured_field[dim:2*dim].copy()

def create_structured(points, vectors):
    return np.vstack([points, vectors])

def create_signed_element(sign, group_element):
    return (sign, group_element)


def scalar_product_structured_unit(unit_structured_field0,
                                   unit_structured_field1, kernel):

    # unit_structured_field are vectors, not matrices (shape = (2*dim,)
    # and not (2*dim, 1) )
    dim = int(unit_structured_field0.shape[0] / 2)

    point0 = unit_structured_field0[0:dim]
    point1 = unit_structured_field1[0:dim]
    vector0 = unit_structured_field0[dim:2*dim]
    vector1 = unit_structured_field1[dim:2*dim]
    return kernel(point0, point1) * np.dot(vector0, vector1)

def generate_scalar_products(structured_field0, structured_field1, kernel, nb_points):
    for i in range(nb_points):
        for j in range(nb_points):
            yield scalar_product_structured_unit(
                    structured_field0[:, i], structured_field1[:, j], kernel)

def scalar_product_structured(structured_field0, structured_field1, kernel):
    nb_points = structured_field0.shape[1]
    scalar_product = sum(generate_scalar_products(structured_field0, structured_field1, kernel, nb_points))
    return scalar_product


def scalar_product_unstructured(structured_field0, field1):
    scalar_product = 0.0
    dim_double, nb_points = structured_field0.shape
    dim = int(dim_double/2)
    points = get_points(structured_field0)
    vectors = get_vectors(structured_field0)

    values_field1 = np.array([
            field1[u].interpolation(points) for u in range(dim)])

    for i in range(nb_points):
        scalar_product += np.dot(values_field1.T[i], vectors[:, i])

    return scalar_product


def get_from_structured_to_unstructured(space, kernel):
    mg = space.meshgrid
    nb_pts_mg0 = mg[0].shape[0]
    nb_pts_mg1 = mg[1].shape[1]
    mg_reshaped = []
    mg_reshaped.append(mg[0].reshape([nb_pts_mg0,1,1]))
    mg_reshaped.append(mg[1].reshape([1,nb_pts_mg1,1]))

    def from_structured_to_unstructured(structured_field):
        dim_double, nb_points = structured_field.shape
        dim = int(dim_double/2)
        points = get_points(structured_field)
        vectors = get_vectors(structured_field)
        unstructured = space.tangent_bundle.zero()
        pt0 = points[0].reshape(1,1,nb_points)
        pt1 = points[1].reshape(1,1,nb_points)
        points_reshaped = [pt0, pt1]
        vectors_reshaped = np.transpose(vectors.reshape(dim,nb_points,1), (0,2,1))
        kern_discr = kernel(mg_reshaped, points_reshaped)
        unstructured = space.tangent_bundle.element([(vectors_reshaped[u] * kern_discr).sum(2) for u in range(dim)])

        return unstructured

    return from_structured_to_unstructured


#
#def get_from_structured_to_unstructured(space, kernel):
#    mg = space.meshgrid
#
#    def from_structured_to_unstructured(structured_field):
#        dim_double, nb_points = structured_field.shape
#        dim = int(dim_double/2)
#        points = get_points(structured_field)
#        vectors = get_vectors(structured_field)
#        unstructured = space.tangent_bundle.zero()
#
#        for k in range(nb_points):
#            def kern_app_point(x):
#                return kernel(x, points[:, k])
#
#            kern_discr = kern_app_point(mg)
#
#            unstructured += space.tangent_bundle.element([kern_discr * vect for vect in vectors[:, k]]).copy()
#
#        return unstructured
#
#    return from_structured_to_unstructured


def make_covariance_matrix(points, kernel):
    """ creates the covariance matrix of the kernel for the given points"""

    dim = len(points)
    p1 = np.reshape(points, (dim, 1, -1))
    p2 = np.reshape(points, (dim, -1, 1))

    return kernel(p1, p2)



def make_covariance_mixte_matrix(points1, points2, kernel):
    """ creates the covariance matrix of the kernel for the given points"""

    dim = len(points1)
    p1 = np.reshape(points1, (dim, -1, 1))
    p2 = np.reshape(points2, (dim, 1, -1))
    
    return kernel(p1, p2)




def get_structured_vectors_from_concatenated(vectors, nb_points, dim):
    return vectors.reshape(nb_points, dim).T

