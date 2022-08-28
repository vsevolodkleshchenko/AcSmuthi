import wavefunctions as wvfs
import tsystem
import numpy as np
# import scipy.special as ss
# import mathematics as mths


class Particle:
    def __init__(self, position, radius, density, speed_of_sound, order):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed = speed_of_sound
        self.incident_field = None
        self.scattered_field = None
        self.inner_filed = None
        self.reflected_field = None
        self.t_matrix = None
        self.d_matrix = None
        self.order = order

    def compute_t_matrix(self, sph_number, ps):
        t = np.zeros(((self.order+1)**2, (self.order+1)**2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            i = n ** 2 + n + m
            t[i, i] = 1 / tsystem.scaled_coefficient(n, sph_number, ps)
        self.t_matrix = t

    def compute_d_matrix(self, ps):
        d = np.zeros(((self.order + 1) ** 2, (self.order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(self.order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(self.order):
                imunu = nu ** 2 + nu + mu
                d[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, ps.k_fluid, self.pos)
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
