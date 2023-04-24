import math
from acsmuthi.utility import mathematics as mths
import numpy as np
import scipy.special
from acsmuthi.initial_field import PlaneWave
from acsmuthi.medium import Medium
from acsmuthi import particles
from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs, fields
'''
test: scattering of plane wave, propagating along z axis, on 1 water sphere.

comparison: solution using package and exact solution using spherical expansion

source: Acoustic force and torque on small objects. Notes. Ivan Toftul
'''


def axisymmetric_outgoing_wvf(n, x, y, z, k):
    r"""Outgoing axisymmetric basis spherical wave function"""
    r, phi, theta = mths.dec_to_sph(x, y, z)
    return mths.spherical_h1n(n, k * r) * scipy.special.lpmv(0, n, np.cos(theta))


def axisymmetric_outgoing_wvfs_array(x, y, z, k, order):
    r"""Builds np.array of all axisymmetric outgoing wave functions with n <= order"""
    as_ow_array = np.zeros((order + 1, *x.shape), dtype=complex)
    for n in range(order + 1):
        as_ow_array[n] = axisymmetric_outgoing_wvf(n, x, y, z, k)
    return as_ow_array


def pn_coefficient_1s(n):
    r"""
    n-th coefficient of expansion plane wave :math:`e^{ikz}`
    in basis of functions :math:`j_n(kr) P_n(\cos\theta)`
    """
    return 1 * 1j ** n * (2 * n + 1)


def re_pn_coefficient_1s(n):
    r"""
    n-th coefficient in decomposition of cross-sections
    """
    return 2 * n + 1


def desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho):
    r"""
    n-th coefficient of expansion of scattered wave (scattering coefficient)
    in basis of functions :math:`h_n^{(1)}(kr) P_n(\cos\theta)`
    """
    gamma = k_s * rho / k / rho_s
    a_n = (gamma * scipy.special.spherical_jn(n, k_s * r_s, derivative=True) *
           scipy.special.spherical_jn(n, k * r_s) - scipy.special.spherical_jn(n, k_s * r_s) *
           scipy.special.spherical_jn(n, k * r_s, derivative=True)) / \
          (scipy.special.spherical_jn(n, k_s * r_s) * mths.spherical_h1n(n, k * r_s, derivative=True) -
           gamma * scipy.special.spherical_jn(n, k_s * r_s, derivative=True) *
           mths.spherical_h1n(n, k * r_s))
    return a_n


def desired_in_coefficient_1s(n, k_s, r_s, rho_s, k, rho):
    r"""
    n-th coefficient of expansion of wave in sphere
    in basis of functions :math:`j_n(kr) P_n(\cos\theta)`
    """
    gamma = k_s * rho / k / rho_s
    c_n = 1j / (k * r_s) ** 2 / (scipy.special.spherical_jn(n, k_s * r_s) * mths.spherical_h1n(n, k * r_s, derivative=True) -
                                 gamma * scipy.special.spherical_jn(n, k_s * r_s, derivative=True) *
                                 mths.spherical_h1n(n, k * r_s))
    return c_n


def desired_pscattered_coefficients_array_1s(k_s, r_s, rho_s, k, rho, shape, order):
    r"""
    array of scattered coefficients for all n <= order,
    repeated length times (number of points) - for field calculation
    """
    sc_coef = np.zeros(order + 1, dtype=complex)
    for n in range(order + 1):
        sc_coef[n] = pn_coefficient_1s(n) * desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho)
    return np.broadcast_to(sc_coef, shape).T


def scattered_field_1s(x, y, z, k_s, r_s, rho_s, k, rho, order):
    r"""
    calculates scattered field in every point
    """
    wave_functions_array = axisymmetric_outgoing_wvfs_array(x, y, z, k, order)
    tot_field_array = desired_pscattered_coefficients_array_1s(k_s, r_s, rho_s, k, rho, wave_functions_array.T.shape, order) * \
                      wave_functions_array
    return np.real(np.sum(tot_field_array, axis=0))


def cross_sections_1s(k_s, r_s, rho_s, k, rho, order):
    r"""
    Calculates scattering and extinction cross-sections with multipoles coefficients
    """
    prefact = 4 * np.pi / k / k
    sigma_sc_array = np.zeros(order + 1)
    sigma_ex_array = np.zeros(order + 1)
    for n in range(order + 1):
        sigma_sc_array[n] = re_pn_coefficient_1s(n) * (np.abs(
            desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho))) ** 2
        sigma_ex_array[n] = re_pn_coefficient_1s(n) * np.real(
            desired_scattered_coefficient_1s(n, k_s, r_s, rho_s, k, rho))
    sigma_sc = prefact * math.fsum(sigma_sc_array) / np.pi / r_s**2
    sigma_ex = - prefact * math.fsum(sigma_ex_array) / np.pi / r_s**2
    return sigma_sc, sigma_ex


def test_one_sphere():
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

    # parameters of grid
    x_p, z_p = np.meshgrid(np.linspace(-6, 6, 201), np.linspace(-6, 6, 201))
    y_p = np.full_like(x_p, 0.)

    incident_field = PlaneWave(amplitude=p0, k_l=k_l, direction=direction)

    fluid = Medium(density=ro_fluid, speed_l=c_fluid)

    sphere1 = particles.SphericalParticle(
        position=pos1,
        radius=r_sph,
        density=ro_sph,
        speed_l=c_sph,
        order=order
    )

    spheres = np.array([sphere1])

    sim = Simulation(
        particles=spheres,
        medium=fluid,
        initial_field=incident_field,
        frequency=freq,
        order=order,
        store_t_matrix=True
    )
    sim.run()

    ecs, scs = cs.cross_section(
        particles=spheres,
        medium=fluid,
        incident_field=incident_field,
        freq=freq,
        order=order
    )

    actual_field = np.real(fields.compute_scattered_field(particles=spheres, x=x_p, y=y_p, z=z_p))

    desired_field = scattered_field_1s(x_p, y_p, z_p, 2 * np.pi * freq / c_sph, r_sph, ro_sph, k_l, ro_fluid, order)

    r = np.sqrt(x_p ** 2 + y_p ** 2 + z_p ** 2)

    actual_field = np.where(r <= sphere1.radius, 0, actual_field)
    desired_field = np.where(r <= sphere1.radius, 0, desired_field)

    err = np.abs((actual_field - desired_field))

    desired_scs, desired_ecs = cross_sections_1s(2 * np.pi * freq / c_sph, r_sph, ro_sph, k_l, ro_fluid, order)

    np.testing.assert_allclose(actual_field, desired_field, rtol=1e-5)
    np.testing.assert_allclose(desired_ecs, ecs)
    np.testing.assert_allclose(desired_scs, scs / (np.pi * r_sph ** 2))
