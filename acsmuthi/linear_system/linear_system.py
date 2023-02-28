import numpy as np

from acsmuthi import fields_expansions as fldsex
import acsmuthi.linear_system.coupling_matrix as cmt
import scipy.special as ss
import scipy.sparse.linalg
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs


class LinearSystem:
    def __init__(self, particles_array, medium, initial_field, frequency, order, store_t_matrix):
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.coupling_matrix = None
        self.particles = particles_array
        self.medium = medium
        self.freq = frequency
        self.incident_field = initial_field
        self.store_t_matrix = store_t_matrix

    def compute_t_matrix(self):
        for sph in range(len(self.particles)):
            self.particles[sph].compute_t_matrix(self.medium.speed_l, self.medium.rho, self.freq)
        self.t_matrix = TMatrix(self.particles, self.order, self.store_t_matrix)

    def compute_coupling_matrix(self):
        self.coupling_matrix = CouplingMatrixExplicit(self.particles, self.order, self.incident_field.k_l)

    def compute_right_hand_side(self):
        rhs = np.zeros((len(self.particles), (self.order + 1) ** 2), dtype=complex)
        for i_p, particle in enumerate(self.particles):
            rhs[i_p] = particle.incident_field.coefficients
        self.rhs = np.concatenate(rhs)

    def prepare(self):
        for particle in self.particles:
            ampl, k_l = self.incident_field.ampl, self.incident_field.k_l
            particle.incident_field = self.incident_field.spherical_wave_expansion(particle.pos, self.order)
            particle.scattered_field = fldsex.SphericalWaveExpansion(ampl, k_l, particle.pos, 'outgoing', self.order)
            if particle.speed_t:
                kst = 2 * np.pi * self.freq / particle.speed_t
            else:
                kst = None
            ksl = 2 * np.pi * self.freq / particle.speed_l
            particle.inner_field = fldsex.SphericalWaveExpansion(ampl, ksl, particle.pos, 'regular', self.order, k_t=kst)
        self.compute_t_matrix()
        self.compute_coupling_matrix()
        self.compute_right_hand_side()

    def solve(self):
        if not self.store_t_matrix:
            master_matrix = self.t_matrix.linear_operator + self.coupling_matrix.linear_operator
            scattered_coefs1d, _ = scipy.sparse.linalg.gmres(master_matrix, self.rhs)
        else:
            master_matrix = self.t_matrix.linear_operator.A + self.coupling_matrix.linear_operator.A
            scattered_coefs1d = scipy.linalg.solve(master_matrix, self.rhs)
        scattered_coefs = scattered_coefs1d.reshape((len(self.particles), (self.order + 1) ** 2))
        inner_coefs = _inner_coefficients(self.particles, scattered_coefs, self.order)
        for s, particle in enumerate(self.particles):
            particle.scattered_field.coefficients = scattered_coefs[s]
            particle.inner_field.coefficients = inner_coefs[s]


class SystemMatrix:
    def __init__(self, particles_array, order):
        self.particles = particles_array
        self.order = order
        self.shape = (len(particles_array) * (order + 1) ** 2, len(particles_array) * (order + 1) ** 2)

    def index_block(self, s):
        return s * (self.order + 1) ** 2


class TMatrix(SystemMatrix):
    def __init__(self, particle_array, order, store_t_matrix):
        SystemMatrix.__init__(self, particle_array, order)

        if not store_t_matrix:
            def apply_t_matrix(vector):
                tv = np.zeros(vector.shape, dtype=complex)
                for i_s, particle in enumerate(particle_array):
                    tv[self.index_block(i_s):self.index_block(i_s + 1)] = particle.t_matrix.dot(
                        vector[self.index_block(i_s):self.index_block(i_s + 1)])
                return tv

            self.linear_operator = scipy.sparse.linalg.LinearOperator(shape=self.shape, matvec=apply_t_matrix,
                                                                      matmat=apply_t_matrix, dtype=complex)
        else:
            t_mat = np.zeros(self.shape, dtype=complex)

            for i_s, particle in enumerate(particle_array):
                t_mat[self.index_block(i_s):self.index_block(i_s + 1),
                      self.index_block(i_s):self.index_block(i_s + 1)] = particle.t_matrix

            self.linear_operator = scipy.sparse.linalg.aslinearoperator(t_mat)


class CouplingMatrixExplicit(SystemMatrix):
    def __init__(self, particle_array, order, k):

        SystemMatrix.__init__(self, particle_array, order)
        coup_mat = np.zeros(self.shape, dtype=complex)

        for sph in range(len(self.particles)):
            other_spheres = np.where(np.arange(len(self.particles)) != sph)[0]
            for osph in other_spheres:
                coup_mat[self.index_block(sph):self.index_block(sph + 1),
                         self.index_block(osph):self.index_block(osph + 1)] = cmt.coupling_block(
                             self.particles[sph].pos, self.particles[osph].pos, k, self.order)

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(coup_mat)


def _inner_coefficients(particles_array, scattered_coefficients, order):
    r"""Counts coefficients of decompositions fields inside spheres"""
    in_coef = np.zeros_like(scattered_coefficients)
    for s, particle in enumerate(particles_array):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            k, k_s = particle.incident_field.k_l, particle.inner_field.k_l
            sc_coef = scattered_coefficients[s, imn]
            in_coef[s, imn] = (ss.spherical_jn(n, k * particle.r) * particle.t_matrix[imn, imn] +
                               mths.sph_hankel1(n, k * particle.r)) * sc_coef/ss.spherical_jn(n, k_s * particle.r)
    return in_coef