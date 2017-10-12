import group
import action
import scipy.optimize as sopt

def calibrate(original, noisy):
    def get_loss(sign):
        def loss(velocity):
            translated = action(group.exponential(velocity), original)
            cov = covariance(translated, translated)
            product = scalar_product(translated, noisy)
            return cov + sign*product
        return loss
    sopt.minimize(loss, group.zero_velocity)
