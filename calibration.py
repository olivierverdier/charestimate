import scipy.optimize as sopt
import numpy as np
import functions_calibration_1D as func_1D

def make_discrepancy_from(noisy, product, pairing):
    """
    Create a discrepancy measure from noisy point,
    using a product and a pairing.
    """
    def discrepancy(translated):
        """
        A measure of the difference between translated and noisy.
        """
        cov = product(translated, translated)
        prod = pairing(translated, noisy)
        return cov - 2*prod
    return discrepancy

def make_loss(original, action, exponential, discrepancy):
    """
    Construct a loss function from a group action, and a discrepancy.
    """
    def loss(velocity):
        """
        Loss function from velocity alone.
        """
        element = exponential(velocity)
        translated = action(element, original)
        return discrepancy(translated)
    return loss

def compute_velocity(original, group, action, discrepancy):
    """
    Minimises discrepancy(exp(xi).v),
    where v is the original.
    """
    loss = make_loss(original, action, group.exponential, discrepancy)

    best =sopt.minimize(loss, group.zero_velocity)
    #best = sopt.minimize(loss, group.zero_velocity)
    return best

def calibrate(original, noisy, group, action, product, pairing):
    """
    Main calibration function.
    """
    discrepancy = make_discrepancy_from(noisy, product, pairing)
    velocity = compute_velocity(original, group, action, discrepancy)
    return velocity

def calibrate_equation_1D_translation(original, noisy, space, kernel):
    """
    'function' is a class that has several method :

     - 'function' which is a function between between the space of vector fields (to which
    noisy belongs) and another vector space W on which the group also acts.
     - 'function_structured' is the same function but acting on structured
    vector fields
     - 'solver' that returns, given w1 and w2, the group element g such that
       w2 = g.w1 or more generally the group element that minimizes
       |w2 - g.w1|_W


    'calibrate_equation' computes calibration thanks to a function that is
    invariant under the group action :
        g.function = g(function(g^-1)) = function \forall g \in g

    """

    function = func_1D.function_1D_translation(space, kernel)
    w_noisy = function.function(noisy)
    w_original = function.function_structured(original)

    g = function.solver(w_original, w_noisy)

    return g


def calibrate_equation(original, noisy, space, kernel, fun_op):
    """
    'function' is a class that has several method :

     - 'function' which is a function between between the space of vector fields (to which
    noisy belongs) and another vector space W on which the group also acts.
     - 'function_structured' is the same function but acting on structured
    vector fields
     - 'solver' that returns, given w1 and w2, the group element g such that
       w2 = g.w1 or more generally the group element that minimizes
       |w2 - g.w1|_W


    'calibrate_equation' computes calibration thanks to a function that is
    invariant under the group action :
        g.function = g(function(g^-1)) = function \forall g \in g

    """

    function = fun_op(space, kernel)
    w_noisy = function.function(noisy)
    w_original = function.function_structured(original)

    g = function.solver(w_original, w_noisy)

    return g


