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
        scalar_product += np.dot(values_field1[:, i], vectors[:, i])

    return scalar_product
