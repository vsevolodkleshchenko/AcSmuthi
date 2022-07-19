import math
import wavefunctions as wvfs
import mathematics as mths
import tsystem
import numpy as np


def total_field(x, y, z, ps, order):
    r""" counts field outside the spheres """
    solution_coefficients = tsystem.solve_system(ps, order)
    tot_field_array = np.zeros((ps.num_sph, (order + 1) ** 2, len(x)), dtype=complex)
    for sph in range(ps.num_sph):
        sphere_solution_coefficients = np.split(np.repeat(solution_coefficients[2 * sph], len(x)), (order + 1) ** 2)
        tot_field_array[sph] = sphere_solution_coefficients * wvfs.outgoing_wave_functions_array(order, x -
                                                                                                 ps.spheres[sph].pos[0],
                                                                                                 y -
                                                                                                 ps.spheres[sph].pos[1],
                                                                                                 z -
                                                                                                 ps.spheres[sph].pos[2],
                                                                                                 ps.k_fluid)
    # tot_field = np.sum(tot_field_array, axis=(0, 1))
    tot_field = mths.spheres_multipoles_fsum(tot_field_array, len(x))
    return tot_field


def cross_section(ps, order):
    r""" Counts scattering and extinction cross sections Sigma_sc and Sigma_ex
    eq(46,47) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    prefactor = - ps.incident_field.ampl ** 2 / (2 * ps.incident_field.omega * ps.fluid.rho * ps.k_fluid)

    solution_coefficients = tsystem.solve_system(ps, order)
    sigma_ex = np.zeros(ps.num_sph * (order + 1) ** 2)
    sigma_sc1 = np.zeros(ps.num_sph * (order + 1) ** 2)
    sigma_sc2 = np.zeros((ps.num_sph * (order + 1) ** 2) ** 2, dtype=complex)
    jmn, jmnlmunu = 0, 0
    for j in range(ps.num_sph):
        for mn in wvfs.multipoles(order):
            for l in np.where(np.arange(ps.num_sph) != j)[0]:
                for munu in wvfs.multipoles(order):
                    sigma_sc2[jmnlmunu] = np.conj(solution_coefficients[2 * j, mn[1] ** 2 + mn[1] + mn[0]]) * \
                                          solution_coefficients[2 * l, munu[1] ** 2 + munu[1] + munu[0]] * \
                                          wvfs.regular_separation_coefficient(munu[0], mn[0], munu[1], mn[1],
                                                                              ps.k_fluid,
                                                                              ps.spheres[j].pos - ps.spheres[l].pos)
                    jmnlmunu += 1
            sigma_sc1[jmn] = np.abs(solution_coefficients[2 * j, mn[1] ** 2 + mn[1] + mn[0]]) ** 2
            sigma_ex[jmn] = - np.real(solution_coefficients[2 * j, mn[1] ** 2 + mn[1] + mn[0]] *
                                      np.conj(wvfs.local_incident_coefficient(mn[0], mn[1], ps.k_fluid,
                                                                              ps.incident_field.dir, ps.spheres[j].pos,
                                                                              order)))
            jmn += 1
    W_sc = (math.fsum(np.real(sigma_sc1)) + math.fsum(np.real(sigma_sc2))) * prefactor
    W_ex = math.fsum(sigma_ex) * prefactor
    sigma_sc = W_sc / ps.intensity_incident_field
    sigma_ex = W_ex / ps.intensity_incident_field
    return sigma_sc, sigma_ex
