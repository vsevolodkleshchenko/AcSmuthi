import numpy as np
import scipy.special as ss

import acsmuthi.utility.mathematics as mths
import acsmuthi.utility.wavefunctions as wvfs


def scaled_coefficient(n, c_medium, rho_medium, c_l, rho, r, freq):
    r"""Scaled coefficient of particle"""
    k_medium = 2 * np.pi * freq / c_medium
    k_l = 2 * np.pi * freq / c_l

    gamma = k_l * rho_medium / k_medium / rho
    s1 = np.zeros((2, 2), dtype=complex)
    s2 = np.zeros((2, 2), dtype=complex)

    s1[0, 0] = gamma * ss.spherical_jn(n, k_medium * r)
    s1[0, 1] = ss.spherical_jn(n, k_l * r)
    s1[1, 0] = ss.spherical_jn(n, k_medium * r, derivative=True)
    s1[1, 1] = ss.spherical_jn(n, k_l * r, derivative=True)

    s2[0, 0] = - gamma * mths.spherical_h1n(n, k_medium * r)
    s2[0, 1] = ss.spherical_jn(n, k_l * r)
    s2[1, 0] = - mths.spherical_h1n(n, k_medium * r, derivative=True)
    s2[1, 1] = ss.spherical_jn(n, k_l * r, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def elastic_scaled_coefficient(n, c_medium, rho_medium, c_l, c_t, rho, r, freq):
    k_l = 2 * np.pi * freq / c_l
    k_t = 2 * np.pi * freq / c_t
    k_medium = 2 * np.pi * freq / c_medium

    sigma = (c_l**2 / 2 - c_t**2) / (c_l**2 - c_t**2)

    alpha_n = ss.spherical_jn(n, k_l * r) - k_l * r * ss.spherical_jn(n, k_l * r, derivative=True)
    beta_n = (n**2 + n - 2) * ss.spherical_jn(n, k_t * r) + k_t ** 2 * r ** 2 ** mths.spherical_jn_der2(n, k_t * r)
    xi_n = k_l * r * ss.spherical_jn(n, k_l * r, derivative=True)
    delta_n = 2 * n * (n + 1) * ss.spherical_jn(n, k_t * r)
    epsilon_n = k_l ** 2 * r ** 2 * (ss.spherical_jn(n, k_l * r) * sigma / (1-2*sigma) - mths.spherical_jn_der2(n,
                                                                                                                k_l * r))
    eta_n = 2 * n * (n + 1) * (ss.spherical_jn(n, k_t * r) - k_t * r * ss.spherical_jn(n, k_t * r, derivative=True))

    coefficient = rho_medium * k_t ** 2 * r ** 2 / 2 / rho
    g_n = coefficient * (alpha_n * delta_n + beta_n * xi_n) / (alpha_n * eta_n + beta_n * epsilon_n)

    scale = - (g_n * ss.spherical_jn(n, k_medium * r) - k_medium * r * ss.spherical_jn(n, k_medium * r, derivative=True)) / \
              (g_n * mths.spherical_h1n(n, k_medium * r) - k_medium * r * mths.spherical_h1n(n, k_medium * r, derivative=True))
    return scale


def t_matrix_sphere(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t=None):
    t = np.zeros(((order+1)**2, (order+1)**2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        i = n ** 2 + n + m
        if c_sphere_t is not None:
            t[i, i] = 1 / elastic_scaled_coefficient(n, c_medium, rho_medium, c_sphere_l, c_sphere_t,
                                                     rho_sphere, r_sphere, freq)
        else:
            t[i, i] = 1 / scaled_coefficient(n, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq)
    return t
