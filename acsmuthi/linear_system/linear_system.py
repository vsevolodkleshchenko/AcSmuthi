from typing import Sequence, Literal

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
    """Process linear system of equations for the scattering problem coefficients.
    """
    def __init__(
        self,
        particles: Sequence[Particle],
        medium: MediumSystem,
        initial_field: InitialField,
        solver: Literal['LU', 'GMRES'] = 'LU',
        use_integration: bool | None = None,    # todo: strange thing
        k_parallel: np.ndarray = None,
    ):
        """Initialize the linear system object.

        Args:
            particles: set of particles in the system
            medium: medium system where the particles are located
            initial_field: incident pressure field
            solver: linear system solver
            use_integration: define whether to use integration for the Sommerfeld coupling matrix
        """
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
        """Assemble particles T-matrices.
        """
        for particle in self.particles:
            particle.compute_t_matrix(
                medium=self.medium.sur_medium,
                frequency=self.incident_field.freq
            )
        self.t_matrix = TMatrix(
            particles=self.particles,
            store_t_matrix=False if self.solver == "GMRES" else True
        )

    def compute_coupling_matrix(self):
        """Assemble coupling matrix.
        """
        if not self._use_integration:
            self.coupling_matrix = CouplingMatrixExplicit(
                particles=self.particles,
                medium=self.medium,
                frequency=self.incident_field.freq
            )
        else:
            self.coupling_matrix = CouplingMatrixSommerfeld(
                particles=self.particles,
                medium=self.medium,
                frequency=self.incident_field.freq,
                k_parallel=self.k_parallel
            )

    def compute_right_hand_side(self):
        """Assemble right-hand side of the linear system.
        """
        rhs = np.zeros(self.t_matrix.shape[0], dtype=complex)
        for i_p, particle in enumerate(self.particles):
            rhs[self.t_matrix.index_block(i_p)] = particle.incident_field.coefficients
        self.rhs = self.t_matrix.linear_operator.matvec(rhs)

    def prepare(self):
        """Prepare the linear system parts for solving.
        """
        amplitude = self.incident_field.amplitude
        freq = self.incident_field.freq
        k = self.medium.sur_medium.wavenumber(freq)

        for particle in self.particles:
            particle.incident_field = self.incident_field.spherical_wave_expansion(
                reference_point=particle.position,
                medium=self.medium,
                order=particle.n_max,
            )
            particle.scattered_field = fldsex.SphericalWaveExpansion(
                amplitude=amplitude,
                k=k,
                reference_point=particle.position,
                kind='outgoing',
                n_max=particle.n_max,
                inner_r=particle.circumscribing_sphere_radius
            )
            particle.inner_field = fldsex.SphericalWaveExpansion(
                amplitude=amplitude,
                k=2 * np.pi * freq / particle.c_longitudinal,
                reference_point=particle.position,
                kind='regular',
                n_max=particle.n_max,
                outer_r=particle.circumscribing_sphere_radius
            )
        self.compute_t_matrix()
        self.compute_coupling_matrix()
        self.compute_right_hand_side()

    def solve(self):
        """Solve the linear system of equations. Evaluate field expansions coefficients.
        """
        master_matrix = MasterMatrix(self.t_matrix, self.coupling_matrix)
        if self.solver == 'GMRES':
            scattered_coefs1d, _ = scipy.sparse.linalg.gmres(master_matrix.linear_operator, self.rhs)
        else:
            scattered_coefs1d = scipy.linalg.solve(master_matrix.linear_operator.A, self.rhs)

        scattered_coefs = scattered_coefs1d.reshape((len(self.particles), -1))
        inner_coefs = _inner_coefficients(
            coupling_matrix=self.coupling_matrix,
            particles=self.particles,
            scattered_coefficients=scattered_coefs,
        )

        for i_p, particle in enumerate(self.particles):
            particle.scattered_field.coefficients = scattered_coefs[i_p]
            particle.inner_field.coefficients = inner_coefs[i_p]


class SystemMatrix:
    """Matrix of the linear system for the scattering problem.
    """
    def __init__(self, particles: Sequence[Particle]):
        """Initialize the system matrix.
        """
        self.particles = particles

        matrix_size = 0
        for particle in particles:
            matrix_size += (particle.n_max + 1) ** 2
        self.shape = (matrix_size, matrix_size)

    def index_block(self, i: int) -> int:
        """Return the indeces correspondent to the i-th particle in the system matrix.
        """
        block_size = (self.particles[i].n_max + 1) ** 2
        return slice(i * block_size, (i + 1) * block_size)


class TMatrix(SystemMatrix):
    """T-matrix of the system. It consists of T-matrices of particles on a diagonal blocks.
    """
    def __init__(
        self,
        particles: Sequence[Particle],
        store_t_matrix: bool
    ):
        """Initialize the T-matrix.

        Args:
            particles: set of particles in the system
            store_t_matrix (bool): whether to store T-matrix in the memory or only dot operator
        """
        SystemMatrix.__init__(self, particles=particles)

        if not store_t_matrix:

            def apply_t_matrix(vector):
                tv = np.zeros(vector.shape, dtype=complex)
                for i_p, particle in enumerate(particles):
                    tv[self.index_block(i_p)] = particle.t_matrix.dot(
                        vector[self.index_block(i_p)]
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

            for i_p, particle in enumerate(particles):
                t_mat[self.index_block(i_p), self.index_block(i_p)] = particle.t_matrix

            self.linear_operator = scipy.sparse.linalg.aslinearoperator(t_mat)


class CouplingMatrixExplicit(SystemMatrix):
    """Explicit coupling matrix for the linear system of equations. No integration is used.
    """
    def __init__(
        self,
        particles: Sequence[Particle],
        medium: MediumSystem,
        frequency: float
    ):
        """Initialize the explicit coupling matrix.

        Args:
            particles: set of particles in the system
            medium: medium system where the particles are located
            frequency: frequency of the fields in Hz
        """
        SystemMatrix.__init__(self, particles=particles)
        self.medium = medium
        self.freq = frequency

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(self.compute_matrix())

    def compute_matrix(self) -> np.ndarray:
        """Compute the coupling matrix blocks for each particles pair and assemble in system matrix.
        """
        coup_mat = np.zeros(self.shape, dtype=complex)

        for i_p, particle_i in enumerate(self.particles):
            for j_p, particle_j in enumerate(self.particles):

                if self.medium.is_substrate:
                    substrate_coupling_block = scmt.substrate_coupling_block(
                        receiver=particle_i,
                        emitter=particle_j,
                        medium=self.medium,
                        frequency=self.freq,
                    )
                    coup_mat[self.index_block(i_p), self.index_block(j_p)] += substrate_coupling_block

                if i_p == j_p:
                    continue

                coup_mat[self.index_block(i_p), self.index_block(j_p)] += cmt.coupling_block(
                    receiver=particle_i,
                    emitter=particle_j,
                    medium=self.medium,
                    frequency=self.freq,
                )

        return coup_mat


class CouplingMatrixSommerfeld(SystemMatrix):
    """Sommerfeld coupling matrix for the linear system of equations.
    For coupling elements evaluation the integration over the k_parallel contour is used.
    """
    def __init__(
        self,
        particles: Sequence[Particle],
        medium: MediumSystem,
        frequency: float,
        k_parallel: np.ndarray | None = None  # todo: make normalized
    ):
        """Initialize the Sommerfeld coupling matrix.

        Args:
            particles: set of particles in the system
            medium: medium system where the particles are located
            frequency: frequency of the fields in Hz
            k_parallel: contour of the unnormalized parallel wavenumber component values for Sommerfeld integrals
        """
        SystemMatrix.__init__(self, particles=particles)
        self.medium = medium
        self.freq = frequency

        if k_parallel is None:
            k = self.medium.sur_medium.wavenumber(self.freq)
            self.k_parallel = scmt.create_default_k_parallel(k, self.medium)
        else:
            self.k_parallel = k_parallel

        self.legendres = self.precompute_legendres(self.k_parallel)

        self.linear_operator = scipy.sparse.linalg.aslinearoperator(self.compute_matrix())

    def compute_matrix(self) -> np.ndarray:
        """Compute the coupling matrix blocks for each particles pair and assemble in system matrix.
        """
        coup_mat = np.zeros(self.shape, dtype=complex)

        for i_p, particle_i in enumerate(self.particles):
            for j_p, particle_j in enumerate(self.particles):

                substrate_coupling_block = scmt.substrate_coupling_block_integrate(
                    receiver=particle_i,
                    emitter=particle_j,
                    medium=self.medium,
                    frequency=self.freq,
                    k_parallel=self.k_parallel,
                    legendres=self.legendres,
                )
                coup_mat[self.index_block(i_p), self.index_block(j_p)] += substrate_coupling_block

                if i_p == j_p:
                    continue

                coup_mat[self.index_block(i_p), self.index_block(j_p)] += cmt.coupling_block(
                    receiver=particle_i,
                    emitter=particle_j,
                    medium=self.medium,
                    frequency=self.freq,
                )

        return coup_mat

    def precompute_legendres(
        self,
        k_parallel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the associated Legendre functions for the given k_parallel contour.
        """
        k = self.medium.sur_medium.wavenumber(self.freq)
        k_z = np.emath.sqrt(k ** 2 - k_parallel ** 2)
        n_max = max(particle.n_max for particle in self.particles)
        return mths.legendres_table(z=k_z / k, n_max=n_max)


class MasterMatrix(SystemMatrix):
    """Master (total) matrix of the linear system of equations.
    """
    def __init__(
        self,
        t_matrix: TMatrix,
        coupling_matrix: CouplingMatrixExplicit
    ):
        """Initialize the master matrix.
        """
        SystemMatrix.__init__(self, particles=t_matrix.particles)
        identity_matrix = np.eye(coupling_matrix.shape[0])
        tw_matrix = t_matrix.linear_operator.matmat(coupling_matrix.linear_operator.A)
        master_matrix = identity_matrix - tw_matrix
        self.linear_operator = scipy.sparse.linalg.aslinearoperator(master_matrix)


def _inner_coefficients(  # todo: maybe delete it / move to the specific particle class
    coupling_matrix: CouplingMatrixExplicit | CouplingMatrixSommerfeld,
    particles: Sequence[Particle],
    scattered_coefficients: np.ndarray
) -> np.ndarray:
    """Counts coefficients of decompositions fields inside spheres
    """
    wc_coefs = coupling_matrix.linear_operator.A @ np.concatenate(scattered_coefficients)
    all_ef_inc_coef = np.split(wc_coefs, len(particles))
    in_coef = np.zeros_like(scattered_coefficients)

    for i_p, particle in enumerate(particles):
        k, k_p = particle.incident_field.k, particle.inner_field.k

        for m, n in wvfs.mn_idx(particle.n_max):
            imn = n ** 2 + n + m

            sc_coef = scattered_coefficients[i_p, imn]
            ef_inc_coef = all_ef_inc_coef[i_p][imn] + particle.incident_field.coefficients[imn]
            jn_ka = ss.spherical_jn(n, k * particle.radius)
            h1n_ka = mths.spherical_h1n(n, k * particle.radius)
            jn_kpa = ss.spherical_jn(n, k_p * particle.radius)
            in_coef[i_p, imn] = (jn_ka * ef_inc_coef + h1n_ka * sc_coef) / jn_kpa

    return in_coef
