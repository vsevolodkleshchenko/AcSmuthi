import numpy as np

from acsmuthi import fields_expansions as fldsex
import acsmuthi.linear_system.coupling_matrix as cmt
import scipy.special as ss
import scipy.sparse.linalg
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import InitialField


class LinearSystem:
    def __init__(
            self,
            particles: np.ndarray[Particle],
            medium: Medium,
            initial_field: InitialField,
            frequency: float,
            order: int,
            store_t_matrix: bool
    ):
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.coupling_matrix = None
        self.particles = particles
        self.medium = medium
        self.freq = frequency
        self.incident_field = initial_field
        self.store_t_matrix = store_t_matrix

    def compute_t_matrix(self):
        for sph in range(len(self.particles)):
            self.particles[sph].compute_t_matrix(
                c_medium=self.medium.cp,
                rho_medium=self.medium.density,
                freq=self.freq
            )
        self.t_matrix = TMatrix(
            particles=self.particles,
            order=self.order,
            store_t_matrix=self.store_t_matrix
        )

    def compute_coupling_matrix(self):
        self.coupling_matrix = CouplingMatrixExplicit(particles=self.particles,
                                                      order=self.order,
                                                      k=self.incident_field.k)

    def compute_right_hand_side(self):
        rhs = np.zeros((len(self.particles), (self.order + 1) ** 2), dtype=complex)
        for i_p, particle in enumerate(self.particles):
            rhs[i_p] = particle.incident_field.coefficients
        self.rhs = np.concatenate(rhs)

    def prepare(self):
        for particle in self.particles:
            amplitude, k = self.incident_field.amplitude, self.incident_field.k
            k_particle = 2 * np.pi * self.freq / particle.cp

            particle.incident_field = self.incident_field.spherical_wave_expansion(
                origin=particle.position,
                order=self.order
            )
            particle.scattered_field = fldsex.SphericalWaveExpansion(
                amplitude=amplitude,
                k=k,
                origin=particle.position,
                kind='outgoing',
                order=self.order,
                inner_r=particle.radius
            )
            particle.inner_field = fldsex.SphericalWaveExpansion(
                amplitude=amplitude,
                k=k_particle,
                origin=particle.position,
                kind='regular',
                order=self.order,
                outer_r=particle.radius
            )
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
    def __init__(
            self,
            particles: np.ndarray[Particle],
            order: int
    ):
        self.particles = particles
        self.order = order
        self.shape = (len(particles) * (order + 1) ** 2, len(particles) * (order + 1) ** 2)

    def index_block(self, s):
        return s * (self.order + 1) ** 2


class TMatrix(SystemMatrix):
    def __init__(
            self,
            particles: np.ndarray[Particle],
            order: int,
            store_t_matrix: bool
    ):
        SystemMatrix.__init__(self, particles=particles, order=order)

        if not store_t_matrix:
            def apply_t_matrix(vector):
                tv = np.zeros(vector.shape, dtype=complex)
                for i_p, particle in enumerate(particles):
                    tv[self.index_block(i_p):self.index_block(i_p + 1)] = particle.t_matrix.dot(
                        vector[self.index_block(i_p):self.index_block(i_p + 1)])
                return tv

            self.linear_operator = scipy.sparse.linalg.LinearOperator(
                shape=self.shape,
                matvec=apply_t_matrix,
                matmat=apply_t_matrix,
                dtype=complex
            )
        else:
            t_mat = np.zeros(self.shape, dtype=complex)

            for i_s, particle in enumerate(particles):
                t_mat[self.index_block(i_s):self.index_block(i_s + 1),
                      self.index_block(i_s):self.index_block(i_s + 1)] = np.linalg.inv(particle.t_matrix)

            self.linear_operator = scipy.sparse.linalg.aslinearoperator(t_mat)


class CouplingMatrixExplicit(SystemMatrix):
    def __init__(
            self,
            particles: np.ndarray[Particle],
            order: int,
            k: float
    ):
        SystemMatrix.__init__(self, particles=particles, order=order)
        coup_mat = np.zeros(self.shape, dtype=complex)

        for sph in range(len(self.particles)):
            other_spheres = np.where(np.arange(len(self.particles)) != sph)[0]
            for osph in other_spheres:
                coup_mat[self.index_block(sph):self.index_block(sph + 1),
                         self.index_block(osph):self.index_block(osph + 1)] = cmt.coupling_block(
                             self.particles[sph].position, self.particles[osph].position, k, self.order)

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(coup_mat)


def _inner_coefficients(particles_array, scattered_coefficients, order):
    r"""Counts coefficients of decompositions fields inside spheres"""
    in_coef = np.zeros_like(scattered_coefficients)
    for i_p, particle in enumerate(particles_array):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            k = particle.incident_field.k
            sc_coef = scattered_coefficients[i_p, imn]
            in_coef[i_p, imn] = (ss.spherical_jn(n, k * particle.radius) / particle.t_matrix[imn, imn] +
                                 mths.spherical_h1n(n, k * particle.radius)) * sc_coef / ss.spherical_jn(n, k * particle.radius)
    return in_coef