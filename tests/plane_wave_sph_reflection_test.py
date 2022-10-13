import numpy as np
from utility import wavefunctions as wvfs, mathematics as mths, reflection
from postprocessing import rendering


def exact_ref_wave(x, y, z):
    ref_direct = reflection.reflection_dir(direct, normal)
    image_o = - 2 * normal * int_dist
    return ref_coef * np.exp(1j * k * ((x - image_o[0]) * ref_direct[0] + (y - image_o[1]) *
                            ref_direct[1] + (z - image_o[2]) * ref_direct[2]))


def image_multipole_ref_wave(x, y, z):
    image_o = - 2 * normal * int_dist
    inc_coefs = wvfs.incident_coefficients(direct, order)
    r_matrix = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(order):
        r_matrix[n**2 + n + m] = (-1) ** (m + n) * ref_coef
    ref_wvfs_array = wvfs.regular_wvfs_array(order, x - image_o[0], y - image_o[1], z - image_o[2], k)
    ref_coefs = r_matrix * inc_coefs
    ref_coefs_array = np.split(np.repeat(ref_coefs, len(x)), (order + 1) ** 2)
    ref_field = mths.multipoles_fsum(ref_coefs_array * ref_wvfs_array, len(x))
    return ref_field


def translation_multipole_ref_wave(x, y, z):
    image_o = - 2 * normal * int_dist
    inc_coefs = wvfs.incident_coefficients(direct, order)
    r_matrix = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(order):
        r_matrix[n**2 + n + m] = (-1) ** (m + n) * ref_coef
    ref_coefs = r_matrix * inc_coefs
    s_matrix = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n**2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu**2 + nu + mu
            s_matrix[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, k, -image_o)
    ref_wvfs_array = wvfs.regular_wvfs_array(order, x, y, z, k)
    ref_coefs_origin = s_matrix @ ref_coefs
    ref_coefs_array = np.split(np.repeat(ref_coefs_origin, len(x)), (order + 1) ** 2)
    ref_field = mths.multipoles_fsum(ref_coefs_array * ref_wvfs_array, len(x))
    return ref_field


c = 331  # [m/s]
freq = 82  # [Hz]
k = 2 * np.pi * freq / c  # [1/m]

direct = np.array([-0.70711, 0, 0.70711])
ref_coef = 0.95
normal = np.array([1, 0, 0])
int_dist = 1

# coordinates
bound, number_of_points = 2.5, 201
span = rendering.build_discretized_span(bound, number_of_points)
plane = 'xz'
plane_number = int(number_of_points / 2) + 1
x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)

# order of decomposition
order = 14


def test():
    actual_field = np.real(image_multipole_ref_wave(x_p - int_dist, y_p, z_p))
    desired_field = np.real(exact_ref_wave(x_p - int_dist, y_p, z_p))
    rendering.plots_for_tests(actual_field, desired_field, span_v, span_h)
    rendering.slice_plot(np.abs(actual_field - desired_field), span_v, span_h)
    np.testing.assert_allclose(actual_field, desired_field, rtol=1e-2)


test()
