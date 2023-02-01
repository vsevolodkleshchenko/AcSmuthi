from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs
import numpy as np


def effective_incident_coefficients(particle):
    r"""Build np.array of effective incident coefficients for all n <= order """
    ef_inc_coef = np.zeros((particle.order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(particle.order):
        imn = n ** 2 + n + m
        ef_inc_coef[imn] = particle.scattered_field.coefficients[imn] * particle.t_matrix[imn, imn]
    return ef_inc_coef


def force_on_sphere(particle, medium):
    r"""Cartesian components of force on sphere[sph]"""
    ef_inc_coef = effective_incident_coefficients(particle)
    scale = particle.t_matrix
    fxy_array = np.zeros((particle.order + 1) ** 2, dtype=complex)
    fz_array = np.zeros((particle.order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(particle.order - 1):
        imn, imn1, imn2 = n ** 2 + n + m, (n + 1) ** 2 + (n + 1) + (m + 1), n ** 2 + n - m
        imn3, imn4 = (n + 1) ** 2 + (n + 1) - (m + 1), (n + 1) ** 2 + (n + 1) + m
        s_coef = 1/scale[imn, imn] + np.conj(1/scale[imn1, imn1]) + 2/scale[imn, imn] * np.conj(1/scale[imn1, imn1])
        coef1 = np.sqrt((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3))
        term1 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1]) + \
                np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
        coef2 = np.sqrt((n - m + 1) * (n + m + 1) / (2 * n + 1) / (2 * n + 3))
        term2 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn4])
        fxy_array[imn], fz_array[imn] = coef1 * term1, coef2 * term2
    k = particle.incident_field.k_l
    prefactor1 = 1j * particle.incident_field.ampl ** 2 / (2 * medium.rho * medium.speed_l ** 2) / 2 / k ** 2
    prefactor2 = particle.incident_field.ampl ** 2 / (2 * medium.rho * medium.speed_l ** 2) / k ** 2
    fxy = prefactor1 * mths.complex_fsum(fxy_array)
    fx, fy = np.real(fxy), np.imag(fxy)
    fz = prefactor2 * np.imag(mths.complex_fsum(fz_array))
    # norm = medium.incident_field.intensity(medium.rho, medium.speed_l) * np.pi * particle.r ** 2 / medium.speed_l
    return np.array([fx, fy, fz])


def all_forces(particles_array, medium):
    r"""Cartesian components of force for all spheres"""
    forces_array = np.zeros((len(particles_array), 3), dtype=float)
    for s, particle in enumerate(particles_array):
        forces_array[s] = force_on_sphere(particle, medium)
    return forces_array
