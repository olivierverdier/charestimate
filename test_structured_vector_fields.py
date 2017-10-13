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
            structured_field0, field, kernel)) == 0

    assert pytest.approx(structured_vector_fields.scalar_product_unstructured(
            structured_field1, field, kernel) == 1)


def test_apply_element_to_field():
    scale = 0
    tx = 0.0
    ty = 0.0
    angle=np.pi/2
    infinitesimal = (scale, (angle, tx, ty))
    infinitesimal1 = (1.0, (angle, tx, ty))
    signed_group_element0=(1, group.exponential(infinitesimal))
    signed_group_element1=(-1, group.exponential(infinitesimal))
    signed_group_element2=(1, group.exponential(infinitesimal1))
    structured_field0 = np.array([[0,0], [0,0], [0,1], [1,0]])

    #expected
    structured_field1 = np.array([[0,0], [0,0], [-1,0], [0,1]])
    structured_field2 = np.array([[0,0], [0,0], [1,0], [0,-1]])
    structured_field3 = np.array([[0,0], [0,0], [-np.exp*(1),0], [0,np.exp*(1)]])

    # obtained
    structured_field4 = structured_vector_fields.apply_element_to_field(signed_group_element0, structured_field0)
    structured_field5 = structured_vector_fields.apply_element_to_field(signed_group_element1, structured_field0)
    structured_field6 = structured_vector_fields.apply_element_to_field(signed_group_element2, structured_field0)

    assert structured_field4.any() == structured_field1.any()

    assert structured_field2.any() == structured_field5.any()

    assert structured_field3.any() == structured_field6.any()


#
#    assert pytest.approx(structured_vector_fields.scalar_product_unstructured(
#            structured_field1, field, kernel) == 1)