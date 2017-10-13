import group
import scipy.optimize as sopt

def get_loss(original, noisy, velocity, sign=1):
    translated = action(group.exponential(velocity), original)
    cov = covariance(translated, translated)
    product = pairing(translated, noisy)
    return cov + 2*sign*product

def calibrate(original, noisy):
    def get_signed_loss(sign):
        def loss(velocity):
            return get_loss(original, noisy, velocity, sign)
        return loss
    best = [sopt.minimize(get_loss(sign), group.zero_velocity)
            for sign in [-1, 1]]
