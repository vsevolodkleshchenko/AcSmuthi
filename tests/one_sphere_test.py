import mathematics as mths
import numpy as np
import scipy
import scipy.special
import rendering
import postprocessing as pp
import wavefunctions as wvfs


def pn_coefficient_1s(n):
    return 1 * 1j ** n * (2 * n + 1)


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
    return pn_coefficient_1s(n) * a_n


def desired_in_coefficient_1s(n, k, ro_fluid, sphere):
    k_abs, k_phi, k_theta = mths.dec_to_sph(k[0], k[1], k[2])
    k_sph, r_sph, ro_sphere = sphere[0], sphere[1], sphere[2]
    gamma = k_sph * ro_fluid / k_abs / ro_sphere
    c_n = 1j / (k_abs * r_sph) ** 2 / \
          (scipy.special.spherical_jn(n, k_sph * r_sph) * mths.sph_hankel1_der(n, k_abs * r_sph) -
           gamma * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True) *
           mths.sph_hankel1(n, k_abs * r_sph))
    return pn_coefficient_1s(n) * c_n


def desired_scattered_coefficients_array_1s(k, ro_fluid, sphere, length, order):
    sc_coef = np.zeros(order, dtype=complex)
    for n in range(order):
        sc_coef[n] = desired_scattered_coefficient_1s(n, k, ro_fluid, sphere)
    return np.split(np.repeat(sc_coef, length), order)


def scattered_field_1s(x, y, z, k, ro_fluid, sphere, order):
    tot_field_array = desired_scattered_coefficients_array_1s(k, ro_fluid, sphere, len(x), order) * \
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

    # plane
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    one_sphere_test(span, plane_number, k, ro_fluid, positions, spheres, order, plane=plane)


one_sphere_simulation()
