from acsmuthi.utility import wavefunctions as wvfs
import numpy as np


def force_on_sphere(particle, medium, initial_field):
    r"""Cartesian components of force on sphere[sph]"""
    ef_inc_coef = np.linalg.inv(particle.t_matrix) @ particle.scattered_field.coefficients
    scale = particle.t_matrix
    fxy_array = np.zeros((particle.order + 1) ** 2, dtype=complex)
    fz_array = np.zeros((particle.order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(particle.order - 1):
        imn, imn1, imn2 = n ** 2 + n + m, (n + 1) ** 2 + (n + 1) + (m + 1), n ** 2 + n - m
        imn3, imn4 = (n + 1) ** 2 + (n + 1) - (m + 1), (n + 1) ** 2 + (n + 1) + m
        s_coef = scale[imn, imn] + np.conj(scale[imn1, imn1]) + 2 * scale[imn, imn] * np.conj(scale[imn1, imn1])
        coef1 = np.sqrt((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3))
        term1 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1]) + \
                np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
        coef2 = np.sqrt((n - m + 1) * (n + m + 1) / (2 * n + 1) / (2 * n + 3))
        term2 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn4])
        fxy_array[imn], fz_array[imn] = coef1 * term1, coef2 * term2
    k = particle.incident_field.k
    prefactor1 = 1j * initial_field.amplitude ** 2 / (2 * medium.density * medium.cp ** 2) / 2 / k ** 2
    prefactor2 = initial_field.amplitude ** 2 / (2 * medium.density * medium.cp ** 2) / k ** 2
    fxy = prefactor1 * np.sum(fxy_array)
    fx, fy = np.real(fxy), np.imag(fxy)
    fz = prefactor2 * np.imag(np.sum(fz_array))
    norm = initial_field.intensity(medium.density, medium.cp) * np.pi * particle.radius ** 2 / medium.cp
    return np.array([fx, fy, fz])  # / norm


def all_forces_old(particles_array, medium, initial_field):
    r"""Cartesian components of force for all spheres"""
    forces_array = np.zeros((len(particles_array), 3), dtype=float)
    for s, particle in enumerate(particles_array):
        forces_array[s] = force_on_sphere(particle, medium, initial_field)
    return forces_array


def all_forces(simulation):
    r"""Cartesian components of force on sphere[sph]"""
    particles, medium, initial_field = simulation.particles, simulation.medium, simulation.initial_field
    forces_array = np.zeros((len(particles), 3), dtype=float)
    scattered_coefficients = np.concatenate([particle.scattered_field.coefficients for particle in particles])
    all_ef_inc_coef = np.split(simulation.linear_system.coupling_matrix.linear_operator.A @ scattered_coefficients, len(particles))
    for s, particle in enumerate(particles):
        ef_inc_coef = - all_ef_inc_coef[s] + particle.incident_field.coefficients
        scale = particle.t_matrix
        fxy_array = np.zeros((particle.order + 1) ** 2, dtype=complex)
        fz_array = np.zeros((particle.order + 1) ** 2, dtype=complex)
        for m, n in wvfs.multipoles(particle.order - 1):
            imn, imn1, imn2 = n ** 2 + n + m, (n + 1) ** 2 + (n + 1) + (m + 1), n ** 2 + n - m
            imn3, imn4 = (n + 1) ** 2 + (n + 1) - (m + 1), (n + 1) ** 2 + (n + 1) + m
            s_coef = scale[imn, imn] + np.conj(scale[imn1, imn1]) + 2 * scale[imn, imn] * np.conj(scale[imn1, imn1])
            coef1 = np.sqrt((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3))
            term1 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1]) + \
                    np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
            coef2 = np.sqrt((n - m + 1) * (n + m + 1) / (2 * n + 1) / (2 * n + 3))
            term2 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn4])
            fxy_array[imn], fz_array[imn] = coef1 * term1, coef2 * term2
        k = particle.incident_field.k
        prefactor1 = 1j * initial_field.amplitude ** 2 / (2 * medium.density * medium.cp ** 2) / 2 / k ** 2
        prefactor2 = initial_field.amplitude ** 2 / (2 * medium.density * medium.cp ** 2) / k ** 2
        fxy = prefactor1 * np.sum(fxy_array)
        fx, fy = np.real(fxy), np.imag(fxy)
        fz = prefactor2 * np.imag(np.sum(fz_array))
        norm = initial_field.intensity(medium.density, medium.cp) * np.pi * particle.radius ** 2 / medium.cp
        forces_array[s] = np.array([fx, fy, fz])  # / norm
    return forces_array
