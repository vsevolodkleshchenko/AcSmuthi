import numpy as np

from acsmuthi.medium import MediumSystem
from acsmuthi.initial_field import InitialField
from acsmuthi.simulation import Simulation
from acsmuthi.utility import wavefunctions as wvfs

# todo: old forces doesn't work properly for substrate - delete


def force_on_sphere(particle, medium: MediumSystem, initial_field: InitialField):
    ef_inc_coef = np.linalg.inv(particle.t_matrix) @ particle.scattered_field.coefficients
    scale = particle.t_matrix

    fxy_array = np.zeros((particle.n_max + 1) ** 2, dtype=complex)
    fz_array = np.zeros((particle.n_max + 1) ** 2, dtype=complex)

    for m, n in wvfs.mn_idx(particle.n_max - 1):
        imn, imn1, imn2 = n ** 2 + n + m, (n + 1) ** 2 + (n + 1) + (m + 1), n ** 2 + n - m
        imn3, imn4 = (n + 1) ** 2 + (n + 1) - (m + 1), (n + 1) ** 2 + (n + 1) + m

        s_coef = scale[imn, imn] + np.conj(scale[imn1, imn1]) + 2 * scale[imn, imn] * np.conj(scale[imn1, imn1])

        coef1 = np.sqrt((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3))
        term11 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1])
        term12 = np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
        term1 = term11 + term12

        coef2 = np.sqrt((n - m + 1) * (n + m + 1) / (2 * n + 1) / (2 * n + 3))
        term2 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn4])

        fxy_array[imn], fz_array[imn] = coef1 * term1, coef2 * term2

    k = medium.sur_medium.wavenumber(initial_field.freq)
    coef = 2 * medium.density * medium.c_longitudinal ** 2
    prefactor1 = 1j * initial_field.amplitude ** 2 / coef / 2 / k ** 2
    prefactor2 = initial_field.amplitude ** 2 / coef / k ** 2

    fxy = prefactor1 * np.sum(fxy_array)
    fx, fy = np.real(fxy), np.imag(fxy)
    fz = prefactor2 * np.imag(np.sum(fz_array))

    # geom_cs = np.pi * particle.radius ** 2
    # norm = initial_field.intensity(medium) * geom_cs / medium.sur_medium.c_longitudinal

    return np.array([fx, fy, fz])  # / norm


def all_forces_old(particles_array, medium, initial_field):
    forces_array = np.zeros((len(particles_array), 3), dtype=float)
    for s, particle in enumerate(particles_array):
        forces_array[s] = force_on_sphere(particle, medium, initial_field)
    return forces_array


def all_forces(simulation: Simulation):
    particles = simulation.particles
    medium = simulation.medium
    initial_field = simulation.initial_field

    forces_array = np.zeros((len(particles), 3), dtype=float)
    scat_cfs = np.concatenate(
        [particle.scattered_field.coefficients for particle in particles]
    )
    wc_coefs = simulation.linear_system.coupling_matrix.linear_operator.A @ scat_cfs
    all_ef_inc_coef = np.split(wc_coefs, len(particles))

    for s, particle in enumerate(particles):
        ef_inc_coef = all_ef_inc_coef[s] + particle.incident_field.coefficients
        scale = particle.t_matrix
        fxy_array = np.zeros((particle.n_max + 1) ** 2, dtype=complex)
        fz_array = np.zeros((particle.n_max + 1) ** 2, dtype=complex)

        for m, n in wvfs.mn_idx(particle.n_max - 1):
            imn, imn1, imn2 = n ** 2 + n + m, (n + 1) ** 2 + (n + 1) + (m + 1), n ** 2 + n - m
            imn3, imn4 = (n + 1) ** 2 + (n + 1) - (m + 1), (n + 1) ** 2 + (n + 1) + m

            s_coef = scale[imn, imn] + np.conj(scale[imn1, imn1]) + 2 * scale[imn, imn] * np.conj(scale[imn1, imn1])

            coef1 = np.sqrt((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3))
            term11 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1])
            term12 = np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
            term1 = term11 + term12

            coef2 = np.sqrt((n - m + 1) * (n + m + 1) / (2 * n + 1) / (2 * n + 3))
            term2 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn4])

            fxy_array[imn], fz_array[imn] = coef1 * term1, coef2 * term2

        k = medium.sur_medium.wavenumber(initial_field.freq)
        coef = 2 * medium.sur_medium.density * medium.sur_medium.c_longitudinal ** 2
        prefactor1 = 1j * initial_field.amplitude ** 2 / coef / 2 / k ** 2
        prefactor2 = initial_field.amplitude ** 2 / coef / k ** 2

        fxy = prefactor1 * np.sum(fxy_array)
        fx, fy = np.real(fxy), np.imag(fxy)
        fz = prefactor2 * np.imag(np.sum(fz_array))

        # geom_cs = np.pi * particle.radius ** 2
        # norm = initial_field.intensity(medium) * geom_cs / medium.sur_medium.c_longitudinal

        forces_array[s] = np.array([fx, fy, fz])  # / norm

    return forces_array
