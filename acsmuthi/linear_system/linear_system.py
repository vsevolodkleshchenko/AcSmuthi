import numpy as np
import scipy.special as ss
import scipy.sparse.linalg

from acsmuthi import fields_expansions as fldsex
import acsmuthi.linear_system.coupling.coupling_matrix as cmt
import acsmuthi.linear_system.coupling.substrate_coupling_matrix as scmt
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
            solver: str,
            use_integration: bool | None = None
    ):
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.coupling_matrix = None
        self.particles = particles
        self.medium = medium
        self.freq = frequency
        self.incident_field = initial_field
        self.solver = solver

        if use_integration is None:
            if medium.is_substrate and not medium.hard_substrate:
                self._use_integration = True
            else:
                self._use_integration = False
        else:
            self._use_integration = use_integration

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
            store_t_matrix=False if self.solver == "GMRES" else True
        )

    def compute_coupling_matrix(self):
        if not self._use_integration:
            self.coupling_matrix = CouplingMatrixExplicit(
                particles=self.particles,
                medium=self.medium,
                order=self.order,
                k=self.incident_field.k
            )
        else:
            self.coupling_matrix = CouplingMatrixSommerfeld(
                particles=self.particles,
                medium=self.medium,
                order=self.order,
                k=self.incident_field.k
            )

    def compute_right_hand_side(self):
        rhs = np.zeros((len(self.particles), (self.order + 1) ** 2), dtype=complex)
        for i_p, particle in enumerate(self.particles):
            rhs[i_p] = particle.incident_field.coefficients
        self.rhs = self.t_matrix.linear_operator.matvec(np.concatenate(rhs))

    def prepare(self):
        for particle in self.particles:
            amplitude, k = self.incident_field.amplitude, self.incident_field.k
            k_particle = 2 * np.pi * self.freq / particle.cp

            particle.incident_field = self.incident_field.spherical_wave_expansion(
                origin=particle.position,
                medium=self.medium,
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
        master_matrix = MasterMatrix(self.t_matrix, self.coupling_matrix)
        if self.solver == 'GMRES':
            scattered_coefs1d, _ = scipy.sparse.linalg.gmres(master_matrix.linear_operator, self.rhs)
        else:
            scattered_coefs1d = scipy.linalg.solve(master_matrix.linear_operator.A, self.rhs)

        scattered_coefs = scattered_coefs1d.reshape((len(self.particles), (self.order + 1) ** 2))
        inner_coefs = _inner_coefficients(self.coupling_matrix, self.particles, scattered_coefs, self.order)

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
                      self.index_block(i_s):self.index_block(i_s + 1)] = particle.t_matrix

            self.linear_operator = scipy.sparse.linalg.aslinearoperator(t_mat)


class CouplingMatrixExplicit(SystemMatrix):
    def __init__(
            self,
            particles: np.ndarray[Particle],
            medium: Medium,
            order: int,
            k: float
    ):
        SystemMatrix.__init__(self, particles=particles, order=order)
        self.medium = medium
        self.k = k

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(self.compute_matrix())

    def compute_matrix(self):
        coup_mat = np.zeros(self.shape, dtype=complex)

        for sph in range(len(self.particles)):
            for osph in range(len(self.particles)):
                if self.medium.is_substrate:
                    substrate_coupling_block = scmt.substrate_coupling_block(
                        self.particles[sph].position, self.particles[osph].position, self.k, self.order)
                    coup_mat[self.index_block(sph):self.index_block(sph + 1),
                             self.index_block(osph):self.index_block(osph + 1)] += substrate_coupling_block

                if sph == osph:
                    continue
                coup_mat[self.index_block(sph):self.index_block(sph + 1),
                         self.index_block(osph):self.index_block(osph + 1)] += cmt.coupling_block(
                    self.particles[sph].position, self.particles[osph].position, self.k, self.order)

        return coup_mat


class CouplingMatrixSommerfeld(SystemMatrix):
    def __init__(
            self,
            particles: np.ndarray[Particle],
            medium: Medium,
            order: int,
            k: float,
            k_parallel: np.ndarray | None = None
    ):
        SystemMatrix.__init__(self, particles=particles, order=order)
        self.medium = medium
        self.k = k
        self.k_parallel = k_parallel

        self.legendres = self.precompute_legendres(k_parallel)

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(self.compute_matrix())

    def compute_matrix(self):
        coup_mat = np.zeros(self.shape, dtype=complex)

        for sph in range(len(self.particles)):
            for osph in range(len(self.particles)):
                substrate_coupling_block = scmt.substrate_coupling_block_integrate(
                    self.particles[sph].position, self.particles[osph].position, self.k, self.order,
                    self.k_parallel, self.legendres, self.medium)
                coup_mat[self.index_block(sph):self.index_block(sph + 1),
                         self.index_block(osph):self.index_block(osph + 1)] += substrate_coupling_block

                if sph == osph:
                    continue
                coup_mat[self.index_block(sph):self.index_block(sph + 1),
                         self.index_block(osph):self.index_block(osph + 1)] += cmt.coupling_block(
                    self.particles[sph].position, self.particles[osph].position, self.k, self.order)

        return coup_mat

    def precompute_legendres(self, k_parallel: str | np.ndarray):
        if k_parallel is None:
            k_p = scmt.k_contour(imag_deflection=1e-2, finish=3, step=1e-2) * self.k
        else:
            k_p = k_parallel
        k_z = np.emath.sqrt(self.k ** 2 - k_p ** 2)
        return mths.legendres_table(k_z / self.k, self.order)


class MasterMatrix(SystemMatrix):
    def __init__(
            self,
            t_matrix: TMatrix,
            coupling_matrix: CouplingMatrixExplicit
    ):
        SystemMatrix.__init__(self, particles=t_matrix.particles, order=t_matrix.order)

        m_mat = np.eye(coupling_matrix.shape[0]) - t_matrix.linear_operator.matmat(coupling_matrix.linear_operator.A)

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(m_mat)


def _inner_coefficients(coupling_matrix, particles_array, scattered_coefficients, order):
    """Counts coefficients of decompositions fields inside spheres"""
    all_ef_inc_coef = np.split(coupling_matrix.linear_operator.A @ np.concatenate(scattered_coefficients), len(particles_array))
    in_coef = np.zeros_like(scattered_coefficients)
    for i_p, particle in enumerate(particles_array):
        k, k_p = particle.incident_field.k, particle.inner_field.k
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            sc_coef = scattered_coefficients[i_p, imn]
            ef_inc_coef = all_ef_inc_coef[i_p][imn] + particle.incident_field.coefficients[imn]
            in_coef[i_p, imn] = (ss.spherical_jn(n, k * particle.radius) * ef_inc_coef +
                                 mths.spherical_h1n(n, k * particle.radius) * sc_coef) / \
                                 ss.spherical_jn(n, k_p * particle.radius)
    return in_coef