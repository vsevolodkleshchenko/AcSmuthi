import numpy as np
import scipy
import scipy.special
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.matrixutils import scipy

import plane_one_sphere

"Everything from 'Acoustic scattering by a pair of spheres' - G.C.Gaunaurd 1995"


def coef_plane_wave(m, n, alpha_x, alpha_y, alpha_z):
    r"""
    d_mn eq(4.40)
    """
    alpha = np.sqrt(alpha_x * alpha_x + alpha_y * alpha_y + alpha_z * alpha_z)
    alpha_phi = np.arccos(alpha_x / alpha)
    alpha_theta = np.arccos(alpha_z / alpha)
    return 4 * np.pi * 1j ** n * scipy.special.sph_harm(m, n, alpha_theta, alpha_phi)


def regular_wvfs(m, n, x, y, z, k):
    r"""
    regular wavefunctions
    """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(x / r)
    theta = np.arccos(z / r)
    return scipy.special.spherical_jn(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def outgoing_wvfs(m, n, x, y, z, k):
    r"""
    outgoing wavefunctions
    """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(x / r)
    theta = np.arccos(z / r)
    return scipy.special.hankel1(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def coef_f(n, k, r_sph, l_sph):
    return scipy.special.spherical_jn(n, k * r_sph, derivative=True) * \
           l_sph * scipy.special.spherical_jn(n, k * r_sph)


# def coef_g(n, k, r_sph, l_sph):
#     return scipy.special.hankel1(n, k * r_sph, derivative=True) * \
#            l_sph * scipy.special.hankel1(n, k * r_sph)
#
#
# def coef_q(n, k, r_sph, l_sph):
#     return coef_f(n, k, r_sph, l_sph) / coef_g(n, k, r_sph, l_sph)
#
#
# def matrix_q(m, k, r_sph, l_sph, order):
#     q = np.eye(order - np.abs(m) + 1)
#     for i in range(np.abs(m), order + 1):
#         q[i, i] = coef_q(i, k, r_sph, l_sph)
#     print(q)
#
#
# matrix_q(3, 1, 1, 1, 7)