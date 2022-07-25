import math
import mathematics as mths
import numpy as np
import scipy.special
import rendering
import postprocessing as pp
import wavefunctions as wvfs
import classes as cls


def pn_coefficient_1s(n):
    return 1 * 1j ** n * (2 * n + 1)


def re_pn_coefficient_1s(n):
    return 2 * n + 1


def desired_scattered_coefficient_1s(n, ps):
    k_sph, r_sph, ro_sphere = ps.k_spheres[0], ps.spheres[0].r, ps.spheres[0].rho
    gamma = k_sph * ps.fluid.rho / ps.k_fluid / ro_sphere
    a_n = (gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           scipy.special.spherical_jn(n, ps.k_fluid * r_sph) - scipy.special.spherical_jn(n, k_sph * r_sph) *
           scipy.special.spherical_jn(n, ps.k_fluid * r_sph, derivative=True)) / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * mths.sph_hankel1_der(n, ps.k_fluid * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           mths.sph_hankel1(n, ps.k_fluid * r_sph))
    return a_n


def desired_in_coefficient_1s(n, ps):
    k_sph, r_sph, ro_sphere = ps.k_spheres[0], ps.spheres[0].r, ps.spheres[0].rho
    gamma = k_sph * ps.fluid.rho / ps.k_fluid / ro_sphere
    c_n = 1j / (ps.k_fluid * r_sph) ** 2 / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * mths.sph_hankel1_der(n, ps.k_fluid * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           mths.sph_hankel1(n, ps.k_fluid * r_sph))
    return c_n


def desired_pscattered_coefficients_array_1s(ps, length, order):
    sc_coef = np.zeros(order + 1, dtype=complex)
    for n in range(order + 1):
        sc_coef[n] = pn_coefficient_1s(n) * desired_scattered_coefficient_1s(n, ps)
    return np.split(np.repeat(sc_coef, length), order + 1)


def scattered_field_1s(x, y, z, ps, order):
    tot_field_array = desired_pscattered_coefficients_array_1s(ps, len(x), order) * \
                      wvfs.axisymmetric_outgoing_wvfs_array(x, y, z, ps.k_fluid, len(x), order)
    return np.sum(tot_field_array, axis=0)


def one_sphere_test(span, plane_number, ps, order, plane='xz'):
    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    desired_1s_field = scattered_field_1s(x_p, y_p, z_p, ps, order)
    actual_1s_field = pp.total_field(x_p, y_p, z_p, ps, order, )
    desired_1s_field = rendering.draw_spheres(desired_1s_field, ps, x_p, y_p, z_p)
    actual_1s_field = rendering.draw_spheres(actual_1s_field, ps, x_p, y_p, z_p)
    rendering.plots_for_tests(actual_1s_field, actual_1s_field - desired_1s_field, span_v, span_h)
    np.testing.assert_allclose(actual_1s_field, desired_1s_field, rtol=1e-5)


def cross_sections_1s(ps, order):
    prefact = 4 * np.pi / ps.k_fluid / ps.k_fluid
    sigma_sc_array = np.zeros(order + 1)
    sigma_ex_array = np.zeros(order + 1)
    for n in range(order + 1):
        sigma_sc_array[n] = re_pn_coefficient_1s(n) * (np.abs(
            desired_scattered_coefficient_1s(n, ps))) ** 2
        sigma_ex_array[n] = re_pn_coefficient_1s(n) * np.real(
            desired_scattered_coefficient_1s(n, ps))
    sigma_sc = prefact * math.fsum(sigma_sc_array)
    sigma_ex = - prefact * math.fsum(sigma_ex_array)
    return sigma_sc, sigma_ex


def one_sphere_simulation():
    # coordinates
    bound, number_of_points = 5, 200
    span = rendering.build_discretized_span(bound, number_of_points)

    ps = cls.build_ps_1s()

    # order of decomposition
    order = 10

    print("Scattering and extinction cross section:", *cross_sections_1s(ps, order))

    # plane
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    one_sphere_test(span, plane_number, ps, order, plane=plane)


one_sphere_simulation()
