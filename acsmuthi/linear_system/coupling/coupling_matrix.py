import numpy as np

from acsmuthi.particles import Particle
from acsmuthi.medium import MediumSystem
import acsmuthi.utility.wavefunctions as wvfs
import acsmuthi.utility.separation_coefficients as seps

# todo: check that it works as expected maybe use numba
try:
    from acsmuthi.utility.cython_opt import cython_speedups as cysp

    def coupling_block(
        receiver: Particle,
        emitter: Particle,
        medium: MediumSystem,
        frequency: float,
    ):
        """Wrapper for coupling matrix block for free space -mediated coupling of two particles.

        Args:
            receiver: particle which receives the scattered field from the emitter
            emitter: particle which emits the scattered field
            medium: medium system where the particles are located
            frequency: frequency of the fields in Hz
        """
        k = medium.sur_medium.wavenumber(frequency)
        order = max(receiver.n_max, emitter.n_max)
        return cysp.coupling_block(receiver.position, emitter.position, k, order)

    def translation_block(order, k_medium, distance):  # todo: maybe delete this function
        return cysp.translation_block(order, k_medium, distance)


except Exception as e:
    print("Failed to import cython speedups", str(e))

    def coupling_block(
        receiver: Particle,
        emitter: Particle,
        medium: MediumSystem,
        frequency: float,
    ) -> np.ndarray:
        """Coupling matrix block for free space -mediated coupling of two particles.

        Args:
            receiver: particle which receives the scattered field from the emitter
            emitter: particle which emits the scattered field
            medium: medium system where the particles are located
            frequency: frequency of the fields in Hz
        """
        shape = (receiver.n_max + 1) ** 2, (emitter.n_max + 1) ** 2
        block = np.zeros(shape, dtype=complex)

        k = medium.sur_medium.wavenumber(frequency)
        distance = receiver.position - emitter.position

        for m, n in wvfs.mn_idx(receiver.n_max):
            imn = n ** 2 + n + m

            for mu, nu in wvfs.mn_idx(emitter.n_max):
                imunu = nu ** 2 + nu + mu

                block[imn, imunu] = seps.outgoing_separation_coefficient(
                    m=mu, mu=m, n=nu, nu=n, k=k, dist=distance
                )
        return block

    def translation_block(order, k_medium, distance):  # todo: maybe delete this function
        d = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
        for m, n in wvfs.mn_idx(order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.mn_idx(order):
                imunu = nu ** 2 + nu + mu
                d[imn, imunu] = seps.regular_separation_coefficient(mu, m, nu, n, k_medium, distance)
        return d
