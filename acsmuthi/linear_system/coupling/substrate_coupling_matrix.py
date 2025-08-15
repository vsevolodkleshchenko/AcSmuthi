import numpy as np
import scipy.special as ss
import scipy.integrate as si

import acsmuthi.utility.wavefunctions as wvfs
from acsmuthi.linear_system.coupling.coupling_basics import k_contour
from acsmuthi.medium import MediumSystem, FluidMedium, ElasticMedium
from acsmuthi.particles import Particle
from acsmuthi.utility.mathematics import car_to_cyl, legendre_prefactor
from acsmuthi.utility.separation_coefficients import gaunt_coefficient


try:
    raise Exception  # todo: check the speedups
    from acsmuthi.utility.cython_opt import cython_speedups as cysp

    def substrate_coupling_block(
        receiver: Particle,
        emitter: Particle,
        medium: MediumSystem,
        frequency: float,
    ) -> np.ndarray:
        """Wrapper for cython implementation of coupling matrix block for substrate-mediated
        coupling of two particles. Evaluated analytically. Suitable only for sound rigid substrate.

        Args:
            receiver: particle which receives the scattered field from the emitter
            emitter: particle which emits the scattered field
            medium: medium system where the particles are located
            frequency: frequency of the fields in Hz
        """
        k = medium.sur_medium.wavenumber(frequency)
        order = max(receiver.n_max, emitter.n_max)
        return cysp.substrate_coupling_block(receiver.position, emitter.position, k, order)


except Exception:

    def substrate_coupling_element(
        m: int, n: int, mu: int, nu: int,
        k: float,
        emitter_pos: np.ndarray,
        receiver_pos: np.ndarray,
    ) -> float:
        """Coupling matrix element for substrate-mediated coupling of two particles through the given
        harmonic numbers. Evaluated analytically. Suitable only for sound rigid substrate.

        Args:
            m: multipole degree of emitted harmonic
            n: multipole order of emitted harmonic
            mu: multipole degree of received harmonic
            nu: multipole order of received harmonic
            k: wavenumber in the medium
            emitter_pos: position of the emitter
            receiver_pos: position of the receiver
        """
        dist = receiver_pos - emitter_pos
        ds = np.abs(emitter_pos[2])

        dx, dy, dz = dist[0], dist[1], dist[2] + 2 * ds

        if abs(n - nu) >= abs(m - mu):
            q0 = abs(n - nu)
        if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
            q0 = abs(m - mu)
        if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
            q0 = abs(m - mu) + 1
        q_lim = (n + nu - q0) // 2

        sum_array = np.zeros(q_lim + 1, dtype=complex)

        for i, q in enumerate(range(0, q_lim + 1)):
            outgoing = wvfs.outgoing_wvf(m - mu, q0 + 2 * q, dx, dy, dz, k)
            gaunt = gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)
            sum_array[i] = 1j ** (q0 + 2 * q) * outgoing * gaunt

        return 4 * np.pi * 1j ** (nu - n) * (-1.) ** (n + m + mu) * np.sum(sum_array)

    def substrate_coupling_block(
        receiver: Particle,
        emitter: Particle,
        medium: MediumSystem,
        frequency: float,
    ) -> np.ndarray:
        """Coupling matrix block for substrate-mediated coupling of two particles.
        Evaluated analytically. Suitable only for sound rigid substrate.

        Args:
            receiver: particle which receives the scattered field from the emitter
            emitter: particle which emits the scattered field
            medium: medium system where the particles are located
            frequency: frequency of the fields in Hz
        """
        shape = (receiver.n_max + 1) ** 2, (emitter.n_max + 1) ** 2
        block = np.zeros(shape, dtype=complex)
        for m, n in wvfs.mn_idx(receiver.n_max):
            imn = n ** 2 + n + m
            for mu, nu in wvfs.mn_idx(emitter.n_max):
                imunu = nu ** 2 + nu + mu
                block[imn, imunu] = substrate_coupling_element(
                    m=mu, n=nu, mu=m, nu=n,
                    k=medium.sur_medium.wavenumber(frequency=frequency),
                    emitter_pos=emitter.position,
                    receiver_pos=receiver.position,
                )
        return block


def substrate_coupling_block_integrate(
    receiver: Particle,
    emitter: Particle,
    medium: MediumSystem,
    frequency: float,
    k_parallel: np.ndarray | float,
    legendres: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:  # todo: if legendres is None - compute them
    """Coupling matrix block for substrate-mediated coupling of two particles.
    Evaluated by integration of Sommerfeld integrals. Suitable for arbitrary substrates.

    Args:
        receiver: particle which receives the scattered field from the emitter
        emitter: particle which emits the scattered field
        medium: medium system where the particles are located
        frequency: frequency of the fields in Hz
        k_parallel: contour of the unnormalized parallel wavenumber component values for Sommerfeld integrals
        legendres: precomputed Legendre polynomials table up to the particles orders
    """
    shape = (receiver.n_max + 1) ** 2, (emitter.n_max + 1) ** 2
    block = np.zeros(shape, dtype=complex)

    receiver_pos = receiver.position
    emitter_pos = emitter.position
    dist = receiver_pos - emitter_pos
    d_rho, d_phi, d_z = car_to_cyl(dist[0], dist[1], dist[2])
    ds = np.abs(emitter_pos[2])

    k = medium.sur_medium.wavenumber(frequency)
    k_z = np.emath.sqrt(k ** 2 - k_parallel ** 2)
    fresnel = medium.fresnel_r(k_parallel=k_parallel, frequency=frequency)

    for m, n in wvfs.mn_idx(receiver.n_max):
        i_mn = n ** 2 + n + m
        leg_mn = legendres[0][m, n] if m >= 0 else legendres[1][-m, n]
        leg_norm_mn = leg_mn * legendre_prefactor(m, n)

        for mu, nu in wvfs.mn_idx(emitter.n_max):
            i_munu = nu ** 2 + nu + mu
            leg_munu = legendres[0][mu, nu] if mu >= 0 else legendres[1][-mu, nu]
            leg_norm_munu = leg_munu * legendre_prefactor(mu, nu)
            leg_norm_munu = leg_norm_munu if (nu + mu) % 2 == 0 else - leg_norm_munu

            eikz = np.exp(1j * k_z * (2 * ds + d_z))
            jn_kp = ss.jn(mu - m, k_parallel * d_rho)
            integrand = fresnel * eikz * k_parallel / k_z * jn_kp * leg_norm_mn * leg_norm_munu
            integral = si.trapz(integrand, k_parallel / k)

            eikp = np.exp(1j * (mu - m) * d_phi)
            block[i_mn, i_munu] = 4 * np.pi * 1j ** (n - nu + mu - m) * eikp * integral
    return block


def create_default_k_parallel(k_medium, medium: MediumSystem):
    frequency = k_medium * medium.sur_medium.c_longitudinal / 2 / np.pi
    if isinstance(medium.substrate, FluidMedium):    # todo: to the MediumSystem
        k_substrate = np.array([medium.substrate.wavenumber(frequency)])
    elif isinstance(medium.substrate, ElasticMedium):
        k_substrate = np.array(
            [medium.substrate.wavenumber(frequency), medium.substrate.wavenumber_transversal(frequency)]
        )
    else:
        k_substrate = None
    if k_substrate is not None:
        branch_points = k_substrate / k_medium
    else:
        branch_points = None
    return k_contour(imag_deflection=1e-2, step=1e-2, problems=branch_points) * k_medium
