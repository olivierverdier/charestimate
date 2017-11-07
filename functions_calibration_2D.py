#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:03:11 2017

@author: bgris
"""

import numpy as np
import structured_vector_fields as struct
import cmath
import group


class function_2D_scalingdisplacement():
    def __init__(self, space, kernel):
        self.space = space
        self.unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)


    def function(self, vect_field):
        """
          returns a list of two vectors : \int v and
          (1 / \int |v| ) * \int x |v(x)|
        """
        dim = 2
        function_one = self.space.one()

        value0 = [function_one.inner(vect_field[i]) for i in range(dim)]

        points = self.space.points().T

        vect_field_abs = self.space.tangent_bundle.element(np.abs(vect_field))
        image_norm = sum([vect_field_abs[i] for i in range(dim)])
        norm1 = function_one.inner(image_norm)

        if (norm1 < 1e-10):
            raise ValueError('problem in function function_2D_scalingdisplacement : norm1 is zero')

        value1 = [(1 / norm1) * function_one.inner(points[i] * image_norm) for i in range(dim)]


        return [value0, value1]

    def function_structured(self, structured_field):
        unstructured_field = self.unstructured_op(structured_field)


        return self.function(unstructured_field)



    def solver(self, w1, w2):
        """
        returns the velocity element such that exponential(g).w1 = w2
        w1 and w2 are lists of 2 lists with 2 elements
        """

        comp1_0 = complex(w1[0][0], w1[0][1])
        comp1_1 = complex(w1[1][0], w1[1][1])
        comp2_0 = complex(w2[0][0], w2[0][1])
        comp2_1 = complex(w2[1][0], w2[1][1])



        norm1 = abs(comp1_0)
        norm2 = abs(comp2_0)

        if norm1 < 1e-10:
            raise ValueError('problem in solver dim1 : norm1 is zero')

        ratio = norm2 / norm1
        # log taken because the result is a velocity, not the group element
        lam = np.log(ratio)

        theta = cmath.phase(comp2_0 / (ratio * comp1_0))

        translation_group = comp2_1 - cmath.rect(1,theta) * comp1_1

        # need to take the 'log' for translation too
        translation = (1 / group.sinc(theta/2)) * cmath.rect(1, - theta / 2) * translation_group

        return np.array([lam, theta, translation.real, translation.imag])







