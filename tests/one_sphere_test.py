import math

import mathematics as mths
import numpy as np
import scipy
import scipy.special
import rendering
import postprocessing as pp
import wavefunctions as wvfs


def pn_coefficient_1s(n):
    return 1 * 1j ** n * (2 * n + 1)


# def pn_coefficient_array_1s(order):
#     pn_c_ar = np.zeros(order + 1, dtype=complex)
#     for n in range(order + 1):
#         pn_c_ar[n] = pn_coefficient_1s(n)
#     return pn_c_ar


def re_pn_coefficient_1s(n):
    return 2 * n + 1


# def re_pn_coefficient_array_1s(order):
#     re_pn_c_ar = np.zeros(order + 1, dtype=complex)
#     for n in range(order + 1):
#         re_pn_c_ar[n] = re_pn_coefficient_1s(n)
#     return re_pn_c_ar


# def dpn_coefficient_array_1s(order):
#     pn_c_ar = np.zeros(order + 1, dtype=complex)
#     for n in range(order + 1):
#         pn_c_ar[n] = 1 * 1j ** n
#     return pn_c_ar


def desired_scattered_coefficient_1s(n, k, ro_fluid, sphere):
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    k_sph, r_sph, ro_sphere = sphere[0], sphere[1], sphere[2]
    gamma = k_sph * ro_fluid / k_abs / ro_sphere
    a_n = (gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           scipy.special.spherical_jn(n, k_abs * r_sph) - scipy.special.spherical_jn(n, k_sph * r_sph) *
           scipy.special.spherical_jn(n, k_abs * r_sph, derivative=True)) / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * mths.sph_hankel1_der(n, k_abs * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           mths.sph_hankel1(n, k_abs * r_sph))
    return a_n


def desired_in_coefficient_1s(n, k, ro_fluid, sphere):
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    k_sph, r_sph, ro_sphere = sphere[0], sphere[1], sphere[2]
    gamma = k_sph * ro_fluid / k_abs / ro_sphere
    c_n = 1j / (k_abs * r_sph) ** 2 / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * mths.sph_hankel1_der(n, k_abs * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           mths.sph_hankel1(n, k_abs * r_sph))
    return c_n


def desired_pscattered_coefficients_array_1s(k, ro_fluid, sphere, length, order):
    sc_coef = np.zeros(order + 1, dtype=complex)
    for n in range(order + 1):
        sc_coef[n] = pn_coefficient_1s(n) * desired_scattered_coefficient_1s(n, k, ro_fluid, sphere)
    return np.split(np.repeat(sc_coef, length), order + 1)


def scattered_field_1s(x, y, z, k, ro_fluid, sphere, order):
    tot_field_array = desired_pscattered_coefficients_array_1s(k, ro_fluid, sphere, len(x), order) * \
                      wvfs.axisymmetric_outgoing_wvf_array(x, y, z, k, len(x), order)
    return np.sum(tot_field_array, axis=0)


def one_sphere_test(span, plane_number, k, ro_fluid, positions, spheres, order, plane='xz'):
    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    desired_1s_field = scattered_field_1s(x_p, y_p, z_p, k, ro_fluid, spheres[0], order)
    actual_1s_field = pp.total_field(x_p, y_p, z_p, k, ro_fluid, positions, spheres, order)
    desired_1s_field = rendering.draw_spheres(desired_1s_field, positions, spheres, x_p, y_p, z_p)
    actual_1s_field = rendering.draw_spheres(actual_1s_field, positions, spheres, x_p, y_p, z_p)
    rendering.plots_for_tests(actual_1s_field, desired_1s_field, span_v, span_h)
    np.testing.assert_allclose(actual_1s_field, desired_1s_field, rtol=2e-2)


def cross_sections_1s(k, ro_fluid, sphere, order):
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    prefact = 4 * np.pi / k_abs / k_abs
    # sigma_inc_array = np.zeros(order + 1)
    sigma_sc_array = np.zeros(order + 1)
    sigma_ex_array = np.zeros(order + 1)
    for n in range(order + 1):
        # sigma_inc_array[n] = re_pn_coefficient_1s(n)
        sigma_sc_array[n] = re_pn_coefficient_1s(n) * (np.abs(desired_scattered_coefficient_1s(n, k, ro_fluid, sphere))) ** 2
        sigma_ex_array[n] = re_pn_coefficient_1s(n) * np.real(desired_scattered_coefficient_1s(n, k, ro_fluid, sphere))
    # sigma_inc = prefact * math.fsum(sigma_inc_array)
    sigma_sc = prefact * math.fsum(sigma_sc_array)
    sigma_ex = - prefact * math.fsum(sigma_ex_array)
    norm = k_abs ** 2
    return sigma_sc * norm, sigma_ex * norm


def one_sphere_simulation():
    # coordinates
    bound, number_of_points = 5, 200
    span = rendering.build_discretized_span(bound, number_of_points)

    freq = 82

    # parameters of fluid
    ro_fluid = 1.225
    c_fluid = 331
    k_fluid = 2 * np.pi * freq / c_fluid

    # parameters of the spheres
    c_sphere = 1403
    k_sphere = 2 * np.pi * freq / c_sphere
    r_sphere = 1
    ro_sphere = 1050
    sphere = np.array([k_sphere, r_sphere, ro_sphere])
    spheres = np.array([sphere])

    # parameters of configuration
    pos = np.array([0, 0, 0])
    positions = np.array([pos])

    # parameters of the field
    k_x = 0
    k_y = 0
    k_z = k_fluid
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 10

    print("Scattering and extinction cross section:", *cross_sections_1s(k, ro_fluid, sphere, order))

    # # plane
    # plane = 'xz'
    # plane_number = int(number_of_points / 2) + 1
    #
    # one_sphere_test(span, plane_number, k, ro_fluid, positions, spheres, order, plane=plane)


one_sphere_simulation()
