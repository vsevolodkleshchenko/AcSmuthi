import wavefunctions as wvfs
import numpy as np
import scipy.special as ss
import mathematics as mths


class Particle:
    def __init__(self, position, radius, density, speed_of_sound, order):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed = speed_of_sound
        self.incident_field = None
        self.scattered_field = None
        self.inner_field = None
        self.reflected_field = None
        self.t_matrix = None
        self.d_matrix = None
        self.order = order

    def compute_t_matrix(self, fluid):
        t = np.zeros(((self.order+1)**2, (self.order+1)**2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            i = n ** 2 + n + m
            t[i, i] = 1 / scaled_coefficient(n, self, fluid)
        self.t_matrix = t

    def compute_d_matrix(self):
        d = np.zeros(((self.order + 1) ** 2, (self.order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(self.order):
                imunu = nu ** 2 + nu + mu
                d[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, self.incident_field.k, self.pos)
        self.d_matrix = d


def coupling_block(particle, other_particle, k_fluid, order):
    block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            distance = particle.pos - other_particle.pos
            block[imn, imunu] = - wvfs.outgoing_separation_coefficient(mu, m, nu, n, k_fluid, distance)
    return block


def scaled_coefficient(n, particle, fluid):
    r"""Scaled coefficient for spheres[sph]"""
    k_q = particle.inner_field.k
    rho_0 = fluid.rho
    k = particle.incident_field.k
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
