import sphrs
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
from matplotlib import colors

# coordinates
number_of_points = 200
l = 5
span_x = np.linspace(-l, l, number_of_points)
span_y = np.linspace(-l, l, number_of_points)
span_z = np.linspace(-l, l, number_of_points)
span = np.array([span_x, span_y, span_z])

# parameters of fluid
freq = 82
ro = 1.225
c_f = 331
k_fluid = 2 * np.pi * freq / c_f

# parameters of the spheres
c_sph = 1403
k_sph = 2 * np.pi * freq / c_sph
r_sph = 2
ro_sph = 1050
sphere = np.array([k_sph, r_sph, ro_sph])
spheres = np.array([sphere])

# parameters of configuration
pos1 = np.array([0, 0, 0])
pos2 = np.array([0, 0, 2.5])
poses = np.array([pos1])

# parameters of the field
k_x = 0  # 0.70711 * k_fluid
k_y = 0
k_z = k_fluid  # 0.70711 * k_fluid
k = np.array([k_x, k_y, k_z])

# order of decomposition
order = 5

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = sphrs.build_slice(span, plane_number, plane=plane)


def plots_for_tests(actual_data, desired_data):
    actual_data = np.real(actual_data).reshape(len(span_v), len(span_h))
    desired_data = np.real(desired_data).reshape(len(span_v), len(span_h))
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(actual_data, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    im2 = ax[1].imshow(desired_data, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    plt.show()


def desired_scattered_coefficient_1s(n):
    gamma = k_sph * ro / k_fluid / ro_sph
    p_n = 1 * 1j ** n * (2 * n + 1)
    a_n = (gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           scipy.special.spherical_jn(n, k_fluid * r_sph) - scipy.special.spherical_jn(n, k_sph * r_sph) *
           scipy.special.spherical_jn(n, k_fluid * r_sph, derivative=True)) / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * sphrs.sph_hankel1_der(n, k_fluid * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           sphrs.sph_hankel1(n, k_fluid * r_sph))
    return p_n * a_n


def desired_in_coefficient_1s(n):
    gamma = k_sph * ro / k_fluid / ro_sph
    p_n = 1 * 1j ** n * (2 * n + 1)
    c_n = 1j / (k_fluid * r_sph) ** 2 / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * sphrs.sph_hankel1_der(n, k_fluid * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           sphrs.sph_hankel1(n, k_fluid * r_sph))
    return p_n * c_n


def desired_scattered_coefficients_array_1s():
    sc_coef = np.zeros(order, dtype=complex)
    for n in range(order):
        sc_coef[n] = desired_scattered_coefficient_1s(n)
    return np.split(np.repeat(sc_coef, len(x_p)), order)


def axisymmetric_regular_wvf(n, x, y, z):
    r, phi, theta = sphrs.dec_to_sph(x, y, z)
    return sphrs.sph_hankel1(n, k_fluid * r) * scipy.special.lpmv(0, n, np.cos(theta))


def axisymmetric_regular_wvf_array(x, y, z):
    as_rw_array = np.zeros((order, len(x_p)), dtype=complex)
    for n in range(order):
        as_rw_array[n] = axisymmetric_regular_wvf(n, x, y, z)
    return as_rw_array


def scattered_field_1s():
    tot_field_array = desired_scattered_coefficients_array_1s() * axisymmetric_regular_wvf_array(x_p, y_p, z_p)
    return np.sum(tot_field_array, axis=0)


def one_sphere_test():
    desired_1s_field = scattered_field_1s()
    actual_1s_field = sphrs.total_field(x_p, y_p, z_p, k, ro, poses, spheres, order)
    desired_1s_field = sphrs.draw_spheres(desired_1s_field, poses, spheres, x_p, y_p, z_p)
    actual_1s_field = sphrs.draw_spheres(actual_1s_field, poses, spheres, x_p, y_p, z_p)
    plots_for_tests(actual_1s_field, desired_1s_field)
    np.testing.assert_allclose(actual_1s_field, desired_1s_field, rtol=1e-2)


one_sphere_test()
