import numpy as np
import scipy.special as ss

import acsmuthi.utility.mathematics as mths
import acsmuthi.utility.wavefunctions as wvfs


def mie_sphere(n, c_medium, rho_medium, c_particle, rho_particle, radius, frequency):  # todo: check the speed w/o det
    """Scattering Mie coefficient of a liquid spherical scatterer.

    :param n: multipole order (non-negative)
    :param c_medium: speed of sound (longitudinal) in surrounding medium
    :param rho_medium: density of surrounding medium
    :param c_particle: speed of sound (longitudinal) in particle
    :param rho_particle: density of particle
    :param radius: radius of spherical particle
    :param frequency: frequency  # todo: maybe go to k
    :return: Mie coefficient of multipole order n
    """
    k_medium = 2 * np.pi * frequency / c_medium
    k_particle = 2 * np.pi * frequency / c_particle

    gamma = k_particle * rho_medium / k_medium / rho_particle
    s1 = np.zeros((2, 2), dtype=complex)
    s2 = np.zeros((2, 2), dtype=complex)

    s1[0, 0] = gamma * ss.spherical_jn(n, k_medium * radius)
    s1[0, 1] = ss.spherical_jn(n, k_particle * radius)
    s1[1, 0] = ss.spherical_jn(n, k_medium * radius, derivative=True)
    s1[1, 1] = ss.spherical_jn(n, k_particle * radius, derivative=True)

    s2[0, 0] = - gamma * mths.spherical_h1n(n, k_medium * radius)
    s2[0, 1] = ss.spherical_jn(n, k_particle * radius)
    s2[1, 0] = - mths.spherical_h1n(n, k_medium * radius, derivative=True)
    s2[1, 1] = ss.spherical_jn(n, k_particle * radius, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def mie_elastic_sphere(n, c_medium, rho_medium, cl_particle, ct_particle, rho_particle, radius, frequency):
    """Scattering Mie coefficient of an elastic spherical scatterer.

    :param n: multipole order (non-negative)
    :param c_medium: speed of sound (longitudinal) in surrounding medium
    :param rho_medium: density of surrounding medium
    :param cl_particle: speed of sound (longitudinal) in particle
    :param ct_particle: speed of sound (transversal) in particle
    :param rho_particle: density of particle
    :param radius: radius of spherical particle
    :param frequency: frequency
    :return: Mie coefficient of multipole order n
    """
    k_l = 2 * np.pi * frequency / cl_particle
    k_t = 2 * np.pi * frequency / ct_particle
    k_medium = 2 * np.pi * frequency / c_medium

    sigma = (cl_particle ** 2 / 2 - ct_particle ** 2) / (cl_particle ** 2 - ct_particle ** 2)

    alpha_n = ss.spherical_jn(n, k_l * radius) - k_l * radius * ss.spherical_jn(n, k_l * radius, derivative=True)
    beta_n = ((n**2 + n - 2) * ss.spherical_jn(n, k_t * radius) +
              k_t ** 2 * radius ** 2 ** mths.spherical_jn_der2(n, k_t * radius))
    xi_n = k_l * radius * ss.spherical_jn(n, k_l * radius, derivative=True)
    delta_n = 2 * n * (n + 1) * ss.spherical_jn(n, k_t * radius)
    epsilon_n = k_l ** 2 * radius ** 2 * (
            ss.spherical_jn(n, k_l * radius) * sigma / (1 - 2 * sigma) - mths.spherical_jn_der2(n, k_l * radius))
    eta_n = 2 * n * (n + 1) * (
            ss.spherical_jn(n, k_t * radius) - k_t * radius * ss.spherical_jn(n, k_t * radius, derivative=True))

    coefficient = rho_medium * k_t ** 2 * radius ** 2 / 2 / rho_particle
    g_n = coefficient * (alpha_n * delta_n + beta_n * xi_n) / (alpha_n * eta_n + beta_n * epsilon_n)

    scale = - (g_n * ss.spherical_jn(n, k_medium * radius) -
               k_medium * radius * ss.spherical_jn(n, k_medium * radius, derivative=True)) / \
            (g_n * mths.spherical_h1n(n, k_medium * radius) -
             k_medium * radius * mths.spherical_h1n(n, k_medium * radius, derivative=True))
    return scale


def t_matrix_sphere(n_max, c_medium, rho_medium, c_particle, rho_particle, radius, frequency):
    """Scattering T-matrix of a liquid spherical object.

    Diagonal matrix consisting of Mie coefficients of a liquid spherical particle.

    :param n_max: maximum multipole order (non-negative)
    :param c_medium: speed of sound (longitudinal) in surrounding medium
    :param rho_medium: density of surrounding medium
    :param c_particle: speed of sound (longitudinal) in particle
    :param rho_particle: density of particle
    :param radius: radius of spherical particle
    :param frequency: frequency
    :return: T-matrix of a sphere
    """
    t = np.zeros(((n_max + 1) ** 2, (n_max + 1) ** 2), dtype=complex)
    for m, n in wvfs.mn_idx(n_max):
        i = n ** 2 + n + m
        t[i, i] = mie_sphere(n, c_medium, rho_medium, c_particle, rho_particle, radius, frequency)
    return t
