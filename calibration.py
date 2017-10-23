import scipy.optimize as sopt


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
    def loss(velocity):
        element = exponential(velocity)
        translated = action(element, original)
        return discrepancy(translated)
    return loss

def compute_velocity(original, group, action, discrepancy):
    loss = make_loss(original, action, group.exponential, discrepancy)
    best = sopt.minimize(loss, group.zero_velocity)
    return best

def calibrate(original, noisy, group, action, product, pairing):
    discrepancy = make_discrepancy_from(noisy, product, pairing)
    velocity = compute_velocity(original, group, action, discrepancy)
    return velocity
