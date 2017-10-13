import group
import structured_vector_fields as struct
import scipy.optimize as sopt


def get_loss(original, noisy, velocity, sign=1, kernel):
    group_element = group.exponential(velocity)
    signed_group_element = struct.create_signed_element(sign, group_element)
    translated = struct.apply_element_to_field(signed_group_element, original)
    cov = struct.scalar_product_structured(translated, translated, kernel)
    product = struct.scalar_product_unstructured(translated, noisy, kernel)
    return cov - 2*sign*product


def calibrate(original, noisy, kernel):
    def get_signed_loss(sign):
        def loss(velocity):
            return get_loss(original, noisy, velocity, sign, kernel)
        return loss
    best = [sopt.minimize(get_loss(sign), group.zero_velocity)
            for sign in [-1, 1]]
