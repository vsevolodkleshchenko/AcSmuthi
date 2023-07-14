import numpy as np
import scipy.special as ss

import acsmuthi.utility.mathematics as mths
import acsmuthi.utility.wavefunctions as wvfs


def t_element(n, c_medium, rho_medium, c_l, rho, r, freq):
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


def t_element_elastic(n, c_medium, rho_medium, c_l, c_t, rho, r, freq):
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


def t_element_elastic1(n, c_medium, rho_medium, c_l, c_t, rho, r, freq):
    omega = 2 * np.pi * freq
    k_l = omega / c_l
    k_s = omega / c_t
    k_medium = omega / c_medium

    e1 = 1j * (k_medium * r) * ss.spherical_jn(n, k_medium * r, derivative=True)
    e2 = -1j * (rho_medium / rho) * k_s ** 2 * r ** 2 * ss.spherical_jn(n, k_medium * r)
    d11 = -1j * (k_medium * r) * mths.spherical_h1n(n, k_medium * r, derivative=True)
    d12 = k_l * r * ss.spherical_jn(n, k_l * r, derivative=True)
    d13 = n * (n + 1) * ss.spherical_jn(n, k_s * r)
    d21 = 1j * (rho_medium / rho) * k_s ** 2 * r ** 2 * mths.spherical_h1n(n, k_medium * r)
    d22 = -4 * k_l * r * ss.spherical_jn(n, k_l * r) + \
        (2 * n * (n + 1) - k_s ** 2 * r ** 2) * ss.spherical_jn(n, k_l * r)
    d23 = 2 * n * (n + 1) * (k_s * r * ss.spherical_jn(n, k_s * r) - ss.spherical_jn(n, k_s * r))
    d32 = 2 * (ss.spherical_jn(n, k_l * r) - k_l * r * ss.spherical_jn(n, k_l * r, derivative=True))
    d33 = 2 * k_s * r * ss.spherical_jn(n, k_s * r, derivative=True) + \
        (k_s ** 2 * r ** 2 - 2 * n * (n + 1) + 2) * ss.spherical_jn(n, k_s * r)

    numerator_matrix = np.zeros((3, 3), dtype=complex)
    denominator_matrix = np.zeros((3, 3), dtype=complex)

    numerator_matrix[0, 0] = e1
    numerator_matrix[1, 0] = e2
    numerator_matrix[0, 1] = d12
    numerator_matrix[1, 1] = d22
    numerator_matrix[2, 1] = d32
    numerator_matrix[0, 2] = d13
    numerator_matrix[1, 2] = d23
    numerator_matrix[2, 2] = d33

    denominator_matrix[0, 0] = d11
    denominator_matrix[1, 0] = d21
    denominator_matrix[0, 1] = d12
    denominator_matrix[1, 1] = d22
    denominator_matrix[2, 1] = d32
    denominator_matrix[0, 2] = d13
    denominator_matrix[1, 2] = d23
    denominator_matrix[2, 2] = d33

    return np.linalg.det(numerator_matrix) / np.linalg.det(denominator_matrix)


def t_matrix_sphere(order, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq, c_sphere_t=None):
    t = np.zeros(((order+1)**2, (order+1)**2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        i = n ** 2 + n + m
        if c_sphere_t is not None:
            t[i, i] = t_element_elastic(n, c_medium, rho_medium, c_sphere_l, c_sphere_t, rho_sphere, r_sphere, freq)
        else:
            t[i, i] = t_element(n, c_medium, rho_medium, c_sphere_l, rho_sphere, r_sphere, freq)
    return t
