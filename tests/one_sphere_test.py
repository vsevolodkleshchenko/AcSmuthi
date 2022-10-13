import math
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs
import numpy as np
import scipy.special
from acsmuthi.fields_expansions import PlaneWave
from acsmuthi.medium import Medium
from acsmuthi import particles
from acsmuthi.linear_system import LinearSystem
from acsmuthi.postprocessing import cross_sections as cs, fields, rendering


def pn_coefficient_1s(n):
    return 1 * 1j ** n * (2 * n + 1)


def re_pn_coefficient_1s(n):
    return 2 * n + 1


def desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho):
    gamma = k_s * rho / k / rho_s
    a_n = (gamma * scipy.special.spherical_jn(n, k_s * r_s, derivative=True) *
           scipy.special.spherical_jn(n, k * r_s) - scipy.special.spherical_jn(n, k_s * r_s) *
           scipy.special.spherical_jn(n, k * r_s, derivative=True)) / \
          (scipy.special.spherical_jn(n, k_s * r_s) * mths.sph_hankel1_der(n, k * r_s) -
           gamma * scipy.special.spherical_jn(n, k_s * r_s, derivative=True) *
           mths.sph_hankel1(n, k * r_s))
    return a_n


def desired_in_coefficient_1s(n, k_s, r_s, rho_s, k, rho):
    gamma = k_s * rho / k / rho_s
    c_n = 1j / (k * r_s) ** 2 / (scipy.special.spherical_jn(n, k_s * r_s) * mths.sph_hankel1_der(n, k * r_s) -
                                 gamma * scipy.special.spherical_jn(n, k_s * r_s, derivative=True) *
                                 mths.sph_hankel1(n, k * r_s))
    return c_n


def desired_pscattered_coefficients_array_1s(k_s, r_s, rho_s, k, rho, length, order):
    sc_coef = np.zeros(order + 1, dtype=complex)
    for n in range(order + 1):
        sc_coef[n] = pn_coefficient_1s(n) * desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho)
    return np.split(np.repeat(sc_coef, length), order + 1)


def scattered_field_1s(x, y, z, k_s, r_s, rho_s, k, rho, order):
    tot_field_array = desired_pscattered_coefficients_array_1s(k_s, r_s, rho_s, k, rho, len(x), order) * \
                      wvfs.axisymmetric_outgoing_wvfs_array(x, y, z, k, len(x), order)
    return np.real(np.sum(tot_field_array, axis=0))


def cross_sections_1s(k_s, r_s, rho_s, k, rho, order):
    prefact = 4 * np.pi / k / k
    sigma_sc_array = np.zeros(order + 1)
    sigma_ex_array = np.zeros(order + 1)
    for n in range(order + 1):
        sigma_sc_array[n] = re_pn_coefficient_1s(n) * (np.abs(
            desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho))) ** 2
        sigma_ex_array[n] = re_pn_coefficient_1s(n) * np.real(
            desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho))
    sigma_sc = prefact * math.fsum(sigma_sc_array)
    sigma_ex = - prefact * math.fsum(sigma_ex_array)
    return sigma_sc, sigma_ex


def one_sphere_test1():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0, 0, 1])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([0, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 8

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = particles.Particle(pos1, r_sph, ro_sph, c_sph, order)
    spheres = np.array([sphere1])

    bound, number_of_points = 6, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    system = LinearSystem(spheres, None, fluid, freq, order)
    system.solve()
    span = rendering.build_discretized_span(bound, number_of_points)
    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    actual_field = np.real(
        fields.compute_total_field(freq, system.medium, system.particles, x_p, y_p, z_p, system.layer))

    desired_field = scattered_field_1s(x_p, y_p, z_p, 2 * np.pi * freq / c_sph, r_sph, ro_sph, k_l, ro_fluid, order)

    r = np.sqrt(x_p ** 2 + y_p ** 2 + z_p ** 2)
    actual_field = np.where(r <= sphere1.r, 0, actual_field)
    desired_field = np.where(r <= sphere1.r, 0, desired_field)

    err = np.abs((actual_field - desired_field))
    rendering.slice_plot(err, span_v, span_h, plane=plane)
    rendering.plots_for_tests(actual_field, desired_field, span_v, span_h)

    print("Actual cs:", *cs.cross_section(system.particles, system.medium, freq, order, None))
    print("Exact cs:", *cross_sections_1s(2 * np.pi * freq / c_sph, r_sph, ro_sph, k_l, ro_fluid, order))

    np.testing.assert_allclose(actual_field, desired_field, rtol=1e-5)


one_sphere_test1()
