import numpy as np
import numpy.linalg
import scipy
import scipy.special
from sympy.physics.wigner import clebsch_gordan
import matplotlib.pyplot as plt
import time



def neyman1(n, z):
    r""" Neyman function of the first kind
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return -1j * scipy.special.hankel1(n, z) + 1j * scipy.special.jv(n, z)


def sph_neyman(n, z):
    r""" spherical Neyman function
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jv(-n - 0.5, z)


def sph_neyman_der(n, z):
    r""" first derivative of spherical Neyman function
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return (-1) ** n * np.sqrt(np.pi / (8 * z ** 3)) * scipy.special.jv(-n - 0.5, z) + \
           (-1) ** (n + 1) * np.sqrt(np.pi / 2 / z) * scipy.special.jvp(-n - 0.5, z)


def sph_hankel1(n, z):
    r""" spherical Hankel function of the first kind
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like """
    return scipy.special.spherical_jn(n, z) + 1j * sph_neyman(n, z)


def sph_hankel1_der(n, z):
    """ first derivative of spherical Hankel function of the first kind
    :param n: array_like - order (float)
    :param z: array_like - argument (float or complex)
    :return: array_like"""
    return scipy.special.spherical_jn(n, z, derivative=True) + 1j * sph_neyman_der(n, z)

def inc_wave_coef(m, n, k_x, k_y, k_z):
    r""" coefficients in decomposition of plane wave
    d^m_n - eq(4.40) of Encyclopedia
    :param m, n: array_like - order and degree of the harmonic (int)
    :param k_x, k_y, k_z: array_like coordinates of incident wave vector
    :return: array_like (complex float) """
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    k_phi = np.arctan(k_y / k_x)
    k_theta = np.arccos(k_z / k)
    return 4 * np.pi * 1j ** n * scipy.special.sph_harm(m, n, k_phi, k_theta)


def regular_wvfs(m, n, x, y, z, k):
    """ regular basis spherical wave functions
    ^psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :param k: array_like - absolute value of incident wave vector
    :return: array_like (complex float) """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(x / y)
    theta = np.arccos(z / r)
    return scipy.special.spherical_jn(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def outgoing_wvfs(m, n, x, y, z, k):
    """outgoing basis spherical wave functions
    psi^m_n - eq(between 4.37 and 4.38) of Encyclopedia
    :param m, n: array_like - order and degree of the wave function(int)
    :param x, y, z: array_like - coordinates
    :param k: array_like - absolute value of incident wave vector
    :return: array_like (complex float) """
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(x / y)
    theta = np.arccos(z / r)
    return sph_hankel1(n, k * r) * \
           scipy.special.sph_harm(m, n, theta, phi)


def gaunt_coef(n, m, nu, mu, q):
    r"""
    Gaunt coefficient: G(n,m;nu,mu;q)
    p.329 in Encyclopedia
    :param n:
    :param m:
    :param nu:
    :param mu:
    :param q:
    :return:
    """
    return np.sqrt((2 * n + 1) * (2 * nu + 1) / (4 * np.pi ) / (2 * q + 1)) * \
           clebsch_gordan(n, 0, nu, 0, q, 0) * clebsch_gordan(n, m, nu, mu, q, m + mu)


def sep_matr_coef(m, mu, n, nu, k, dist_x, dist_y, dist_z, order):
    """coefficient S^mmu_nnu(b) of separation matrix
    eq(3.86) in Encyclopedia
    :param m: array_like - number of coefficient(int)
    :param n: array_like - number of coefficient(int)
    :param nu: array_like - number of coefficient(int)
    :param k: array_like - absolute value of incident wave vector
    :param dist: float - distance between 2 spheres
    :return: array_like"""
    dist = np.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z)
    dist_phi = np.arctan(dist_x / dist_y)
    dist_theta = np.arccos(dist_z . dist)
    sum = 0
    for q in range(order):
        sum += 1j ** q * sph_hankel1(q, k * dist) * \
               np.conj(scipy.special.sph_harm(mu - m, q, dist_theta, dist_phi)) * \
               gaunt_coef(n, m, q, mu - m, nu)
    return 4 * np.pi * 1j ** (nu - n) * sum


def t_matrix(k, ro, k_sph, r_sph, ro_sph, order):
    num_of_coef = (order + 1) ** 2
    t = np.zeros((num_of_coef * 2, num_of_coef * 2), dtype=complex)
    for i in range(num_of_coef):
        t[2 * i, i] = - sph_hankel1(i, k * r_sph)
        t[2 * i + 1, i] = - sph_hankel1_der(i, k * r_sph)
        j = i + num_of_coef
        t[2 * i, j] = scipy.special.spherical_jn(i, k_sph * r_sph)
        t[2 * i + 1, j] = ro / ro_sph * scipy.special.spherical_jn(i, k_sph * r_sph, derivative=True)
    return t


def decomp_coef(k_x, k_y, k_z, ro, k_sph, r_sph, ro_sph, order):
    k = np.sqrt(k_x * k_x + k_y * k_y + k_z * k_z)
    num_of_coef = ((order + 1) ** 2) * 2
    b = np.zeros(num_of_coef, dtype=complex)
    for n in range(order):
        idx = np.arange(2 * (n ** 2), 2 * (n + 1) ** 2)
        m = np.arange(-n, n + 1)
        m2 = np.concatenate(np.array([m, m]).T)
        b[idx] = inc_wave_coef(m2, n, k_x, k_y, k_z)
    t = t_matrix(k, ro, k_sph, r_sph, ro_sph, order)
    c = np.linalg.solve(t, b)


decomp_coef(2, 3, 1, 2, 3, 4, 5, 2)
# print(t_matrix(3,1,2,6,4,3))