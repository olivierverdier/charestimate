import numpy as np
import structured_vector_fields as struct
import cmath

class function_1D_translation():
    def __init__(self, space, kernel):
        self.space = space
        self.extent = space.max_pt[0] -space.min_pt[0]
        self.k = 1
        self.unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)

    def get_cos_fun(self, k):
        return self.space.tangent_bundle.element([[np.cos((2 / self.extent) *self.k * np.pi * x)
                    for x in self.space.points()]])

    def get_sin_fun(self, k):
        return self.space.tangent_bundle.element([[np.sin((2 / self.extent) *self.k * np.pi * x)
                    for x in self.space.points()]])

    def function(self, vect_field):
        fun_cos = self.get_cos_fun(self.k)
        fun_sin = self.get_sin_fun(self.k)
        return complex(vect_field.inner(fun_cos), vect_field.inner(fun_sin))

    def function_structured(self, structured_field):
        unstructured_field = self.unstructured_op(structured_field)
        k=self.k
        fun_cos = self.get_cos_fun(k)
        fun_sin = self.get_sin_fun(k)
        result = complex(unstructured_field.inner(fun_cos), unstructured_field.inner(fun_sin))


        return result


#        while (abs(result) < 1e-10 and k < 100):
#            k += 1
#            fun_cos = self.get_cos_fun(k)
#            fun_sin = self.get_sin_fun(k)
#            result = complex(unstructured_field.inner(fun_cos), unstructured_field.inner(fun_sin))
#
#        self.k = k

    def solver(self, w1, w2):
        """
        returns the velocity element such that exponential(g).w1 = w2
        w1 and w2 are complex numbers
        """

        norm1 = abs(w1)
        #norm2 = abs(w2)

        if norm1 < 1e-10:
            raise ValueError('problem in solver dim1 : norm1 is zero')

        #lam = norm2 / norm1
        translation = cmath.phase(w2 / w1) * self.extent / (2 * np.pi)

        return np.array([translation])


class function_1D_scalingtranslation():
    def __init__(self, space, kernel):
        self.space = space
        self.extent = space.max_pt[0] -space.min_pt[0]
        self.k = 1
        self.unstructured_op = struct.get_from_structured_to_unstructured(space, kernel)

    def get_cos_fun(self, k):
        return self.space.tangent_bundle.element([[np.cos((2 / self.extent) *self.k * np.pi * x)
                    for x in self.space.points()]])

    def get_sin_fun(self, k):
        return self.space.tangent_bundle.element([[np.sin((2 / self.extent) *self.k * np.pi * x)
                    for x in self.space.points()]])

    def function(self, vect_field):
        fun_cos = self.get_cos_fun(self.k)
        fun_sin = self.get_sin_fun(self.k)
        return complex(vect_field.inner(fun_cos), vect_field.inner(fun_sin))

    def function_structured(self, structured_field):
        unstructured_field = self.unstructured_op(structured_field)
        k=self.k
        fun_cos = self.get_cos_fun(k)
        fun_sin = self.get_sin_fun(k)
        result = complex(unstructured_field.inner(fun_cos), unstructured_field.inner(fun_sin))

        return result

#
#        while (abs(result) < 1e-10 and k < 100):
#            k += 1
#            fun_cos = self.get_cos_fun(k)
#            fun_sin = self.get_sin_fun(k)
#            result = complex(unstructured_field.inner(fun_cos), unstructured_field.inner(fun_sin))
#
#        self.k = k


    def solver(self, w1, w2):
        """
        returns the velocity element such that exponential(g).w1 = w2
        w1 and w2 are complex numbers
        """

        norm1 = abs(w1)
        norm2 = abs(w2)

        if norm1 < 1e-10:
            raise ValueError('problem in solver dim1 : norm1 is zero')

        # log taken because the result is a velocity, not the group element
        lam = np.log(norm2 / norm1)
        translation = cmath.phase(w2 / w1) * self.extent / (2 * np.pi)

        return np.array([[lam], [translation]])







