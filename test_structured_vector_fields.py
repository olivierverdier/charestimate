#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:48:04 2017

@author: bgris
"""

import structured_vector_fields
import group
import pytest
import numpy as np
import odl
import numpy.testing as npt




def test_scalar_product_structured_unit():
    def kernel(x,y):
        return 1

    unit_structured_field0 = np.array([0,0,1,0])
    unit_structured_field1 = np.array([0,0,0,1])

    assert pytest.approx(structured_vector_fields.scalar_product_structured_unit(
            unit_structured_field0, unit_structured_field1, kernel)) == 0

    assert pytest.approx(structured_vector_fields.scalar_product_structured_unit(
            unit_structured_field0, unit_structured_field1, kernel) == 1)

def test_scalar_product_structured():
    def kernel(x,y):
        return 1

    structured_field0 = np.array([[0,0], [0,0], [1,1], [0,0]])
    structured_field1 = np.array([[0,0], [0,0], [0,0], [1,1]])

    assert pytest.approx(structured_vector_fields.scalar_product_structured(
            structured_field0, structured_field1, kernel)) == 0

    assert pytest.approx(structured_vector_fields.scalar_product_structured(
            structured_field0, structured_field1, kernel) == 4)


def test_scalar_product():
    def kernel(x,y):
        return 1

    structured_field0 = np.array([[0,0], [0,0], [1,1], [0,0]])
    structured_field1 = np.array([[0,0], [0,0], [0,1], [1,0]])
    space = odl.uniform_discr(
    min_pt =[-10,-10], max_pt=[10,10], shape=[8,8],
        dtype='float32', interp='linear')

    field=space.tangent_bundle.element([space.zero(), space.one()])

    assert pytest.approx(structured_vector_fields.scalar_product_unstructured(
            structured_field0, field)) == 0

    assert pytest.approx(structured_vector_fields.scalar_product_unstructured(
            structured_field1, field) == 1)

def test_make_covariance_matrix():
    def kernel(x,y):
        return np.sum(x + y, axis = 0)

    dim = 2
    nb_points = 3

    points = np.random.randn(dim, nb_points)
    Mat_expected = np.empty([nb_points, nb_points])

    for i in range(nb_points):
        for j in range(nb_points):
            Mat_expected[i, j] = kernel(points[:, i], points[:, j])

    Mat_computed = structured_vector_fields.make_covariance_matrix(points, kernel)

    npt.assert_allclose(Mat_computed, Mat_expected)
