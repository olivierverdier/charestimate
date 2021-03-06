import numpy as np
import odl
import structured_vector_fields as struct
import calibration as calib
import group
import action



def get_kernel(space):
    proj = projection_periodicity(space)

    scale = 0.5*space.cell_sides
    #vol_cell = space.cell_volume
    def kernel(point0, point1):

        point0_periodic =proj(point0)
        point1_periodic = proj(point1)

        #value = 0.0
        mask = np.abs(point0_periodic - point1_periodic.T) < scale

        result = np.zeros_like(mask, dtype=float)
        result[mask] = 1.0

        return result

    return kernel



def get_kernel_gauss(space, sigma):
    proj = group.projection_periodicity(space)
    def kernel(point0, point1):

        point0_periodic =proj(point0)
        point1_periodic = proj(point1)
        return np.exp(- sum([ (xi - yi) ** 2 for xi, yi in zip(point0_periodic,point1_periodic)]) / (sigma ** 2))

    return kernel

def test_calibration():

    space = odl.uniform_discr(
            min_pt =[-1], max_pt=[1], shape=[128],
            dtype='float32', interp='linear')

    cell_side = space.cell_sides

    #kernel = get_kernel(space)
    kernel = get_kernel_gauss(space, 0.2)
    def product(f, g):
        return struct.scalar_product_structured(f, g, kernel)

    #points = space.points()[::2].T
    points = np.array([[-0.75, 0.0, 0.2, 0.5,]])
    vectors = np.array([[0.3, 0.0, 0, 1,]])
    original = struct.create_structured(points, vectors)
    g = group.Translation(space)

    translation = np.array([1.0])
    translated = action.apply_element_to_field(g, translation, original)



    covariance_matrix = struct.make_covariance_matrix(space.points().T, kernel)
    noise_l2 =  odl.phantom.noise.white_noise(space)*0.05
    decomp = np.linalg.cholesky(covariance_matrix + 1e-5 * np.identity(len(covariance_matrix)))
    noise_rkhs = np.dot(decomp, noise_l2)

    get_unstructured = struct.get_from_structured_to_unstructured(space, kernel)
    noisy = space.tangent_bundle.element(get_unstructured(translated) + noise_rkhs)


    def act(element, struct):
        return action.apply_element_to_field(g, element, struct)
    result_calibration = calib.calibrate(original, noisy, g, act, product, struct.scalar_product_unstructured)

    estimated_translated = get_unstructured(act(result_calibration.x, original))
    print('real = {}, computed ={} , log diff = {}'.format(translation, result_calibration.x, np.log10(np.abs(translation[0] - result_calibration.x[0]))))






