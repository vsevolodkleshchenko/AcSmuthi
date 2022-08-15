import wavefunctions as wvfs
import mathematics as mths
import tsystem
import numpy as np


def force_on_sphere(sph, sc_coef, ps, order):
    r""" return cartesian components of force on sphere[sph] - eq(34, 35) - lopes2016 """
    ef_inc_coef = tsystem.effective_incident_coefficients(sph, sc_coef, ps, order)
    scale = tsystem.scaled_coefficient
    fxy_array = np.zeros((order + 1) ** 2, dtype=complex)
    fz_array = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(order - 1):
        imn, imn1, imn2 = n ** 2 + n + m, (n + 1) ** 2 + (n + 1) + (m + 1), n ** 2 + n - m
        imn3, imn4 = (n + 1) ** 2 + (n + 1) - (m + 1), (n + 1) ** 2 + (n + 1) + m
        s_coef = scale(n, sph, ps) + np.conj(scale(n + 1, sph, ps)) + \
                 2 * scale(n, sph, ps) * np.conj(scale(n+1, sph, ps))
        coef1 = np.sqrt((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3))
        term1 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1]) + \
               np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
        coef2 = np.sqrt((n - m + 1) * (n + m + 1) / (2 * n + 1) / (2 * n + 3))
        term2 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn4])
        fxy_array[imn], fz_array[imn] = coef1 * term1, coef2 * term2
    prefactor1 = 1j * ps.incident_field.ampl ** 2 / (2 * ps.fluid.rho * ps.fluid.speed ** 2) / 2 / ps.k_fluid ** 2
    prefactor2 = ps.incident_field.ampl ** 2 / (2 * ps.fluid.rho * ps.fluid.speed ** 2) / ps.k_fluid ** 2
    fxy = prefactor1 * mths.complex_fsum(fxy_array)
    fx, fy = np.real(fxy), np.imag(fxy)
    fz = prefactor2 * np.imag(mths.complex_fsum(fz_array))
    return np.array([fx, fy, fz])


def all_forces(solution_coefficients, ps, order):
    r""" return np.array of cartesian components of force for all spheres """
    forces_array = np.zeros((ps.num_sph, 3), dtype=float)
    for sph in range(ps.num_sph):
        forces_array[sph] = force_on_sphere(sph, solution_coefficients[1][sph], ps, order)
    return forces_array
