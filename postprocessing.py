import math
import wavefunctions as wvfs
import mathematics as mths
import tsystem
import numpy as np


def total_field(x, y, z, solution_coefficients, ps, order, incident_field=False):
    r""" counts field outside the spheres """
    scattered_coefficients = solution_coefficients[0]
    sc_field_array = np.zeros((ps.num_sph, (order + 1) ** 2, len(x)), dtype=complex)
    for sph in range(ps.num_sph):
        sph_sc_coef = np.split(np.repeat(scattered_coefficients[sph], len(x)), (order + 1) ** 2)
        sc_field_array[sph] = wvfs.outgoing_wvfs_array(order, x - ps.spheres[sph].pos[0], y - ps.spheres[sph].pos[1],
                                                        z - ps.spheres[sph].pos[2], ps.k_fluid) * sph_sc_coef
    # tot_field = np.sum(tot_field_array, axis=(0, 1))
    tot_field = mths.spheres_multipoles_fsum(sc_field_array, len(x))
    if incident_field:
        inc_field_array = wvfs.incident_coefficients_array(ps.incident_field.dir, len(x), order) * \
                          wvfs.regular_wvfs_array(order, x, y, z, ps.k_fluid)
        inc_field = mths.multipoles_fsum(inc_field_array, len(x))
        tot_field += inc_field
    return tot_field


def cross_section(solution_coefficients, ps, order):
    r""" Counts scattering and extinction cross sections Sigma_sc and Sigma_ex
    eq(46,47) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    prefactor = ps.incident_field.ampl ** 2 / (2 * ps.incident_field.omega * ps.fluid.rho * ps.k_fluid)
    scattered_coefficients = solution_coefficients[0]
    sigma_ex = np.zeros(ps.num_sph * (order + 1) ** 2)
    sigma_sc1 = np.zeros(ps.num_sph * (order + 1) ** 2)
    sigma_sc2 = np.zeros((ps.num_sph * (order + 1) ** 2) ** 2, dtype=complex)
    idx1, idx2 = 0, 0
    for sph in range(ps.num_sph):
        for mn in wvfs.multipoles(order):
            for osph in np.where(np.arange(ps.num_sph) != sph)[0]:
                for munu in wvfs.multipoles(order):
                    sigma_sc2[idx2] = np.conj(scattered_coefficients[sph, mn[1] ** 2 + mn[1] + mn[0]]) * \
                                      scattered_coefficients[osph, munu[1] ** 2 + munu[1] + munu[0]] * \
                                      wvfs.regular_separation_coefficient(munu[0], mn[0], munu[1], mn[1], ps.k_fluid,
                                                                          ps.spheres[sph].pos - ps.spheres[osph].pos)
                    idx2 += 1
            sigma_sc1[idx1] = np.abs(scattered_coefficients[sph, mn[1] ** 2 + mn[1] + mn[0]]) ** 2
            sigma_ex[idx1] = - np.real(scattered_coefficients[sph, mn[1] ** 2 + mn[1] + mn[0]] *
                                       np.conj(wvfs.local_incident_coefficient(mn[0], mn[1], ps.k_fluid,
                                                                              ps.incident_field.dir, ps.spheres[sph].pos,
                                                                              order)))
            idx1 += 1
    W_sc = (math.fsum(sigma_sc1) + math.fsum(np.real(sigma_sc2))) * prefactor
    W_ex = math.fsum(sigma_ex) * prefactor
    sigma_sc = W_sc / ps.intensity_incident_field
    sigma_ex = W_ex / ps.intensity_incident_field
    return sigma_sc, sigma_ex


def force_on_sphere(sph, sc_coef, ps, order):
    r""" return cartesian components of force on sphere[sph] - eq(34, 35) - lopes2016 """
    ef_inc_coef = tsystem.effective_incident_coefficients(sph, sc_coef, ps, order)
    scale = tsystem.scaled_coefficient
    fxy_array = np.zeros((order + 1) ** 2, dtype=complex)
    fz_array = np.zeros((order + 1) ** 2, dtype=complex)
    for mn in wvfs.multipoles(order - 1):
        imn = mn[1] ** 2 + mn[1] + mn[0]
        imn1 = (mn[1] + 1) ** 2 + (mn[1] + 1) + (mn[0] + 1)
        imn2 = mn[1] ** 2 + mn[1] - mn[0]
        imn3 = (mn[1] + 1) ** 2 + (mn[1] + 1) - (mn[0] + 1)
        imn4 = (mn[1] + 1) ** 2 + (mn[1] + 1) + mn[0]
        s_coef = scale(mn[1], sph, ps) + np.conj(scale(mn[1] + 1, sph, ps)) + \
                 2 * scale(mn[1], sph, ps) * np.conj(scale(mn[1]+1, sph, ps))
        coef1 = np.sqrt((mn[1] + mn[0] + 1) * (mn[1] + mn[0] + 2) / (2 * mn[1] + 1) / (2 * mn[1] + 3))
        term1 = s_coef * ef_inc_coef[imn] * np.conj(ef_inc_coef[imn1]) + \
               np.conj(s_coef) * np.conj(ef_inc_coef[imn2]) * ef_inc_coef[imn3]
        coef2 = np.sqrt((mn[1] - mn[0] + 1) * (mn[1] + mn[0] + 1) / (2 * mn[1] + 1) / (2 * mn[1] + 3))
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
        forces_array[sph] = force_on_sphere(sph, solution_coefficients[0][sph], ps, order)
    return forces_array
