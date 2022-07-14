import math
import wavefunctions as wvfs
import mathematics as mths
import tsystem
import numpy as np


def total_field(x, y, z, k, ro, pos, spheres, order):
    r""" counts field outside the spheres """
    solution_coefficients = tsystem.solve_system(k, ro, pos, spheres, order)
    tot_field_array = np.zeros((len(spheres), (order + 1) ** 2, len(x)), dtype=complex)
    for sph in range(len(spheres)):
        sphere_solution_coefficients = np.split(np.repeat(solution_coefficients[2 * sph], len(x)), (order + 1) ** 2)
        tot_field_array[sph] = sphere_solution_coefficients * wvfs.outgoing_wave_functions_array(order, x - pos[sph][0],
                                                                                                 y - pos[sph][1],
                                                                                                 z - pos[sph][2], k)
    # tot_field = np.sum(tot_field_array, axis=(0, 1))
    tot_field = mths.spheres_multipoles_fsum(tot_field_array, len(x))
    return tot_field


def cross_section(k, ro, pos, spheres, order):
    r""" Counts scattering and extinction cross sections Sigma_sc and Sigma_ex
    eq(46,47) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    coef = tsystem.solve_system(k, ro, pos, spheres, order)
    num_sph = len(spheres)
    sigma_ex = np.zeros(num_sph * (order + 1) ** 2)
    sigma_sc1 = np.zeros(num_sph * (order + 1) ** 2)
    sigma_sc2 = np.zeros((num_sph * (order + 1) ** 2) ** 2, dtype=complex)
    jmn, jmnlmunu = 0, 0
    for j in range(num_sph):
        for mn in wvfs.multipoles(order):
            for l in np.where(np.arange(num_sph) != j)[0]:
            # for l in range(num_sph):
                for munu in wvfs.multipoles(order):
                    sigma_sc2[jmnlmunu] = np.conj(coef[2 * j, mn[1] ** 2 + mn[1] + mn[0]]) * \
                               coef[2 * l, munu[1] ** 2 + munu[1] + munu[0]] * \
                                          wvfs.regular_separation_coefficient(munu[0], mn[0], munu[1], mn[1], k,
                                                                         pos[j] - pos[l])
                    jmnlmunu += 1
            sigma_sc1[jmn] = np.abs(coef[2 * j, mn[1] ** 2 + mn[1] + mn[0]]) ** 2
            sigma_ex[jmn] = - np.real(coef[2 * j, mn[1] ** 2 + mn[1] + mn[0]] *
                                      np.conj(wvfs.local_incident_coefficient(mn[0], mn[1], k, pos[j], order)))
            jmn += 1
    sigma_sc = math.fsum(np.real(sigma_sc1)) + math.fsum(np.real(sigma_sc2))
    sigma_ex = math.fsum(sigma_ex)
    return sigma_sc, sigma_ex


# needs revision
def total_field_m(x, y, z, k, ro, pos, spheres, order, m=-1):
    r""" Counts field outside the spheres for mth harmonic """
    coef = tsystem.solve_system(k, ro, pos, spheres, order)
    tot_field = 0
    for n in range(abs(m), order + 1):
        for sph in range(len(spheres)):
            tot_field += coef[2 * sph][n ** 2 + n + m] * \
                         wvfs.outgoing_wave_function(m, n, x - pos[sph][0], y - pos[sph][1], z - pos[sph][2], k)
        tot_field += wvfs.incident_coefficient(m, n, k) * wvfs.regular_wave_function(m, n, x, y, z, k)
    return tot_field