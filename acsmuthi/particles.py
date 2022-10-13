import numpy as np
import scipy.special as ss
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs


class Particle:
    def __init__(self, position, radius, density, speed_of_sound, order, speed_t=None):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed_l = speed_of_sound
        self.speed_t = speed_t
        self.incident_field = None
        self.scattered_field = None
        self.inner_field = None
        self.reflected_field = None
        self.t_matrix = None
        self.d_matrix = None
        self.order = order

    def compute_t_matrix(self, c_medium, rho_medium, freq):
        t = np.zeros(((self.order+1)**2, (self.order+1)**2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            i = n ** 2 + n + m
            if self.speed_t:
                t[i, i] = 1 / elastic_scaled_coefficient(n, c_medium, rho_medium, self.speed_l, self.speed_t, self.rho,
                                                         self.r, freq)
            else:
                t[i, i] = 1 / scaled_coefficient(n, c_medium, rho_medium, self.speed_l, self.rho, self.r, freq)
        self.t_matrix = t

    def compute_d_matrix(self):
        d = np.zeros(((self.order + 1) ** 2, (self.order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(self.order):
                imunu = nu ** 2 + nu + mu
                d[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, self.incident_field.k_l, self.pos)
        self.d_matrix = d


def coupling_block(particle_pos, other_particle_pos, k_medium, order):
    block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            distance = particle_pos - other_particle_pos
            block[imn, imunu] = - wvfs.outgoing_separation_coefficient(mu, m, nu, n, k_medium, distance)
    return block


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

    s2[0, 0] = - gamma * mths.sph_hankel1(n, k_medium * r)
    s2[0, 1] = ss.spherical_jn(n, k_l * r)
    s2[1, 0] = - mths.sph_hankel1_der(n, k_medium * r)
    s2[1, 1] = ss.spherical_jn(n, k_l * r, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def elastic_scaled_coefficient(n, c_medium, rho_medium, c_l, c_t, rho, r, freq):
    k_l = 2 * np.pi * freq / c_l
    k_t = 2 * np.pi * freq / c_t
    k_medium = 2 * np.pi * freq / c_medium

    sigma = (c_l**2 / 2 - c_t**2) / (c_l**2 - c_t**2)

    alpha_n = ss.spherical_jn(n, k_l * r) - k_l * r * ss.spherical_jn(n, k_l * r, derivative=True)
    beta_n = (n**2 + n - 2) * ss.spherical_jn(n, k_t * r) + k_t ** 2 * r ** 2 ** mths.sph_bessel_der2(n, k_t * r)
    xi_n = k_l * r * ss.spherical_jn(n, k_l * r, derivative=True)
    delta_n = 2 * n * (n + 1) * ss.spherical_jn(n, k_t * r)
    epsilon_n = k_l ** 2 * r ** 2 * (ss.spherical_jn(n, k_l * r) * sigma/(1-2*sigma) - mths.sph_bessel_der2(n, k_l * r))
    eta_n = 2 * n * (n + 1) * (ss.spherical_jn(n, k_t * r) - k_t * r * ss.spherical_jn(n, k_t * r, derivative=True))

    coefficient = rho_medium * k_t ** 2 * r ** 2 / 2 / rho
    g_n = coefficient * (alpha_n * delta_n + beta_n * xi_n) / (alpha_n * eta_n + beta_n * epsilon_n)

    scale = - (g_n * ss.spherical_jn(n, k_medium * r) - k_medium * r * ss.spherical_jn(n, k_medium * r, derivative=True)) / \
            (g_n * mths.sph_hankel1(n, k_medium * r) - k_medium * r * mths.sph_hankel1_der(n, k_medium * r))
    return scale

