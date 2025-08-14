from typing import Sequence

import numpy as np
import scipy.special as ss
import scipy.sparse.linalg

from acsmuthi import fields_expansions as fldsex
import acsmuthi.linear_system.coupling.coupling_matrix as cmt
import acsmuthi.linear_system.coupling.substrate_coupling_matrix as scmt
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs
from acsmuthi.particles import Particle
from acsmuthi.medium import MediumSystem, RigidBoundary
from acsmuthi.initial_field import InitialField


class LinearSystem:     # todo: think about CUDA, logging, tqdm, saving?
    def __init__(
            self,
            particles: Sequence[Particle],
            medium: MediumSystem,
            initial_field: InitialField,
            order: int,
            solver: str,
            use_integration: bool | None = None,    # todo: strange thing
            k_parallel: np.ndarray = None,
    ):
        self.order = order
        self.rhs = None
        self.t_matrix = None
        self.coupling_matrix = None
        self.particles = particles
        self.medium = medium
        self.incident_field = initial_field
        self.solver = solver
        self.k_parallel = k_parallel

        if use_integration is None:
            if medium.is_substrate and not isinstance(medium.substrate, RigidBoundary):
                self._use_integration = True
            else:
                self._use_integration = False
        else:
            self._use_integration = use_integration

    def compute_t_matrix(self):
        for sph in range(len(self.particles)):
            self.particles[sph].compute_t_matrix(
                medium=self.medium.sur_medium,
                frequency=self.incident_field.freq
            )
        self.t_matrix = TMatrix(
            particles=self.particles,
            order=self.order,
            store_t_matrix=False if self.solver == "GMRES" else True
        )

    def compute_coupling_matrix(self):
        k = self.medium.sur_medium.wavenumber(self.incident_field.freq)
        if not self._use_integration:
            self.coupling_matrix = CouplingMatrixExplicit(
                particles=self.particles,
                medium=self.medium,
                order=self.order,
                k=k
            )
        else:
            self.coupling_matrix = CouplingMatrixSommerfeld(
                particles=self.particles,
                medium=self.medium,
                order=self.order,
                k=k,
                k_parallel=self.k_parallel
            )

    def compute_right_hand_side(self):
        rhs_shape = (len(self.particles), (self.order + 1) ** 2)
        rhs = np.zeros(rhs_shape, dtype=complex)
        for i_p, particle in enumerate(self.particles):
            rhs[i_p] = particle.incident_field.coefficients
        self.rhs = self.t_matrix.linear_operator.matvec(np.concatenate(rhs))

    def prepare(self):
        amplitude = self.incident_field.amplitude
        freq = self.incident_field.freq
        k = self.medium.sur_medium.wavenumber(freq)

        for particle in self.particles:
            particle.incident_field = self.incident_field.spherical_wave_expansion(
                reference_point=particle.position,
                medium=self.medium,
                order=self.order
            )
            particle.scattered_field = fldsex.SphericalWaveExpansion(
                amplitude=amplitude,
                k=k,
                reference_point=particle.position,
                kind='outgoing',
                n_max=self.order,
                inner_r=particle.circumscribing_sphere_radius
            )
            particle.inner_field = fldsex.SphericalWaveExpansion(
                amplitude=amplitude,
                k=2 * np.pi * freq / particle.c_longitudinal,
                reference_point=particle.position,
                kind='regular',
                n_max=self.order,
                outer_r=particle.circumscribing_sphere_radius
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
        inner_coefs = _inner_coefficients(
            coupling_matrix=self.coupling_matrix,
            particles_array=self.particles,
            scattered_coefficients=scattered_coefs,
            order=self.order
        )

        for s, particle in enumerate(self.particles):
            particle.scattered_field.coefficients = scattered_coefs[s]
            particle.inner_field.coefficients = inner_coefs[s]


class SystemMatrix:
    def __init__(self, particles: Sequence[Particle], order: int):
        self.particles = particles
        self.order = order
        self.shape = (len(particles) * (order + 1) ** 2, len(particles) * (order + 1) ** 2)

    def index_block(self, s):
        return s * (self.order + 1) ** 2


class TMatrix(SystemMatrix):
    def __init__(
        self,
        particles: Sequence[Particle],
        order: int,
        store_t_matrix: bool
    ):
        SystemMatrix.__init__(self, particles=particles, order=order)

        if not store_t_matrix:
            def apply_t_matrix(vector):
                tv = np.zeros(vector.shape, dtype=complex)
                for i_p, particle in enumerate(particles):
                    tv[self.index_block(i_p):self.index_block(i_p + 1)] = particle.t_matrix.dot(
                        vector[self.index_block(i_p):self.index_block(i_p + 1)]
                    )
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
                t_mat[
                    self.index_block(i_s):self.index_block(i_s + 1),
                    self.index_block(i_s):self.index_block(i_s + 1)
                ] = particle.t_matrix

            self.linear_operator = scipy.sparse.linalg.aslinearoperator(t_mat)


class CouplingMatrixExplicit(SystemMatrix):
    def __init__(
        self,
        particles: Sequence[Particle],
        medium: MediumSystem,
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
                        receiver_pos=self.particles[sph].position,
                        emitter_pos=self.particles[osph].position,
                        k=self.k,
                        order=self.order
                    )
                    coup_mat[
                        self.index_block(sph):self.index_block(sph + 1),
                        self.index_block(osph):self.index_block(osph + 1)
                    ] += substrate_coupling_block

                if sph == osph:
                    continue
                coup_mat[
                    self.index_block(sph):self.index_block(sph + 1),
                    self.index_block(osph):self.index_block(osph + 1)
                ] += cmt.coupling_block(
                    particle_pos=self.particles[sph].position,
                    other_particle_pos=self.particles[osph].position,
                    k_medium=self.k,
                    order=self.order
                )

        return coup_mat


class CouplingMatrixSommerfeld(SystemMatrix):
    def __init__(
            self,
            particles: Sequence[Particle],
            medium: MediumSystem,
            order: int,
            k: float,
            k_parallel: np.ndarray | None = None
    ):
        SystemMatrix.__init__(self, particles=particles, order=order)
        self.medium = medium
        self.k = k

        if k_parallel is None:
            self.k_parallel = scmt.create_default_k_parallel(self.k, self.medium)
        else:
            self.k_parallel = k_parallel

        self.legendres = self.precompute_legendres(self.k_parallel)

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(self.compute_matrix())

    def compute_matrix(self):
        coup_mat = np.zeros(self.shape, dtype=complex)

        for sph in range(len(self.particles)):
            for osph in range(len(self.particles)):
                substrate_coupling_block = scmt.substrate_coupling_block_integrate(     # todo: arguments as objects
                    receiver_pos=self.particles[sph].position,
                    emitter_pos=self.particles[osph].position,
                    k=self.k,
                    order=self.order,
                    k_parallel=self.k_parallel,
                    legendres=self.legendres,
                    medium=self.medium
                )
                coup_mat[
                    self.index_block(sph):self.index_block(sph + 1),
                    self.index_block(osph):self.index_block(osph + 1)
                ] += substrate_coupling_block

                if sph == osph:
                    continue
                coup_mat[
                    self.index_block(sph):self.index_block(sph + 1),
                    self.index_block(osph):self.index_block(osph + 1)
                ] += cmt.coupling_block(
                    particle_pos=self.particles[sph].position,
                    other_particle_pos=self.particles[osph].position,
                    k_medium=self.k,
                    order=self.order
                )

        return coup_mat

    def precompute_legendres(self, k_parallel: np.ndarray):
        k_z = np.emath.sqrt(self.k ** 2 - k_parallel ** 2)
        return mths.legendres_table(k_z / self.k, self.order)


class MasterMatrix(SystemMatrix):
    def __init__(
        self,
        t_matrix: TMatrix,
        coupling_matrix: CouplingMatrixExplicit
    ):
        SystemMatrix.__init__(self, particles=t_matrix.particles, order=t_matrix.order)
        identity_matrix = np.eye(coupling_matrix.shape[0])
        tw_matrix = t_matrix.linear_operator.matmat(coupling_matrix.linear_operator.A)
        master_matrix = identity_matrix - tw_matrix
        self.linear_operator = scipy.sparse.linalg.aslinearoperator(master_matrix)


def _inner_coefficients(coupling_matrix, particles_array, scattered_coefficients, order):
    # todo: maybe we should delete it / move to the specific particle class
    """Counts coefficients of decompositions fields inside spheres"""
    wc_coefs = coupling_matrix.linear_operator.A @ np.concatenate(scattered_coefficients)
    all_ef_inc_coef = np.split(wc_coefs, len(particles_array))
    in_coef = np.zeros_like(scattered_coefficients)
    for i_p, particle in enumerate(particles_array):
        k, k_p = particle.incident_field.k, particle.inner_field.k
        for m, n in wvfs.mn_idx(order):
            imn = n ** 2 + n + m
            sc_coef = scattered_coefficients[i_p, imn]
            ef_inc_coef = all_ef_inc_coef[i_p][imn] + particle.incident_field.coefficients[imn]
            jn_ka = ss.spherical_jn(n, k * particle.radius)
            h1n_ka = mths.spherical_h1n(n, k * particle.radius)
            jn_kpa = ss.spherical_jn(n, k_p * particle.radius)
            in_coef[i_p, imn] = (jn_ka * ef_inc_coef + h1n_ka * sc_coef) / jn_kpa
    return in_coef
