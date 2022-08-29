import numpy as np
import scipy.special as ss
from utility import mathematics as mths, wavefunctions as wvfs


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

    def compute_t_matrix(self, medium):
        t = np.zeros(((self.order+1)**2, (self.order+1)**2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            i = n ** 2 + n + m
            if self.speed_t:
                t[i, i] = 1 / elastic_scaled_coefficient(n, self, medium)
            else:
                t[i, i] = 1 / scaled_coefficient(n, self, medium)
        self.t_matrix = t

    def compute_d_matrix(self):
        d = np.zeros(((self.order + 1) ** 2, (self.order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(self.order):
                imunu = nu ** 2 + nu + mu
                d[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, self.incident_field.k_l, self.pos)
        self.d_matrix = d


def coupling_block(particle, other_particle, k_medium, order):
    block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            distance = particle.pos - other_particle.pos
            block[imn, imunu] = - wvfs.outgoing_separation_coefficient(mu, m, nu, n, k_medium, distance)
    return block


def scaled_coefficient(n, particle, medium):
    r"""Scaled coefficient of particle"""
    k_q = particle.inner_field.k_l
    rho_0 = medium.rho
    k = particle.incident_field.k_l
    rho_q = particle.rho
    a_q = particle.r
    gamma_q = k_q * rho_0 / k / rho_q
    s1 = np.zeros((2, 2), dtype=complex)
    s2 = np.zeros((2, 2), dtype=complex)

    s1[0, 0] = gamma_q * ss.spherical_jn(n, k * a_q)
    s1[0, 1] = ss.spherical_jn(n, k_q * a_q)
    s1[1, 0] = ss.spherical_jn(n, k * a_q, derivative=True)
    s1[1, 1] = ss.spherical_jn(n, k_q * a_q, derivative=True)

    s2[0, 0] = - gamma_q * mths.sph_hankel1(n, k * a_q)
    s2[0, 1] = ss.spherical_jn(n, k_q * a_q)
    s2[1, 0] = - mths.sph_hankel1_der(n, k * a_q)
    s2[1, 1] = ss.spherical_jn(n, k_q * a_q, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def elastic_scaled_coefficient(n, particle, medium):
    k = particle.incident_field.k_l
    k_l = particle.inner_field.k_l
    k_t = particle.inner_field.k_t
    a = particle.r
    c_l = particle.speed_l
    c_t = particle.speed_t

    sigma = (c_l**2 / 2 - c_t**2) / (c_l**2 - c_t**2)

    alpha_n = ss.spherical_jn(n, k_l * a) - k_l * a * ss.spherical_jn(n, k_l * a, derivative=True)
    beta_n = (n**2 + n - 2) * ss.spherical_jn(n, k_t * a) + k_t**2 * a**2 ** mths.sph_bessel_der2(n, k_t * a)
    xi_n = k_l * a * ss.spherical_jn(n, k_l * a, derivative=True)
    delta_n = 2 * n * (n + 1) * ss.spherical_jn(n, k_t * a)
    epsilon_n = k_l**2 * a**2 * (ss.spherical_jn(n, k_l * a) * sigma/(1-2*sigma) - mths.sph_bessel_der2(n, k_l * a))
    eta_n = 2 * n * (n + 1) * (ss.spherical_jn(n, k_t * a) - k_t * a * ss.spherical_jn(n, k_t * a, derivative=True))

    coefficient = medium.rho * k_t ** 2 * a ** 2 / 2 / particle.rho
    g_n = coefficient * (alpha_n * delta_n + beta_n * xi_n) / (alpha_n * eta_n + beta_n * epsilon_n)

    scale = - (g_n * ss.spherical_jn(n, k * a) - k * a * ss.spherical_jn(n, k * a, derivative=True)) /\
              (g_n * mths.sph_hankel1(n, k * a) - k * a * mths.sph_hankel1_der(n, k * a))
    return scale

