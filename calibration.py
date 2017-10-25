import scipy.optimize as sopt
import numpy as np

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
    def loss_array(velocity_array):
        velocity_element= (velocity_array[0], (velocity_array[1], velocity_array[2], velocity_array[3]))
        return loss(velocity_element)
    best =sopt.minimize(loss_array, np.array([0,0,0,0]))
    #best = sopt.minimize(loss, group.zero_velocity)
    return best

def calibrate(original, noisy, group, action, product, pairing):
    """
    Main calibration function.
    """
    discrepancy = make_discrepancy_from(noisy, product, pairing)
    velocity = compute_velocity(original, group, action, discrepancy)
    return velocity
