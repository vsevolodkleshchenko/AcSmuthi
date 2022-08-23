import math
import wavefunctions as wvfs
import numpy as np


def scattering_cs(solution_coefficients, ps, order):
    r"""Counts scattering cross section"""
    prefactor = ps.incident_field.ampl ** 2 / (2 * ps.incident_field.omega * ps.fluid.rho * ps.k_fluid)
    scattered_coefficients = solution_coefficients[1]
    sigma_sc2 = np.zeros((ps.num_sph * (order + 1) ** 2) ** 2, dtype=complex)
    idx2 = 0
    for sph in range(ps.num_sph):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for osph in np.where(np.arange(ps.num_sph) != sph)[0]:
                for mu, nu in wvfs.multipoles(order):
                    imunu = nu ** 2 + nu + mu
                    sigma_sc2[idx2] = np.conj(scattered_coefficients[sph, imn]) * scattered_coefficients[osph, imunu] *\
                                      wvfs.regular_separation_coefficient(mu, m, nu, n, ps.k_fluid,
                                                                          ps.spheres[sph].pos - ps.spheres[osph].pos)
                    idx2 += 1
    sigma_sc1 = math.fsum(np.concatenate(np.abs(scattered_coefficients)) ** 2)
    W_sc = (sigma_sc1 + math.fsum(np.real(sigma_sc2))) * prefactor
    sigma_sc = W_sc / ps.intensity_incident_field
    return sigma_sc


def extinction_cs(solution_coefficients, ps):
    r"""Counts extinction cross section"""
    prefactor = ps.incident_field.ampl ** 2 / (2 * ps.incident_field.omega * ps.fluid.rho * ps.k_fluid)
    local_inc_coefs, scattered_coefficients = solution_coefficients[0], solution_coefficients[1]
    if ps.interface:
        local_inc_coefs += solution_coefficients[4]
    sigma_ex = - math.fsum(np.real(np.concatenate(scattered_coefficients) * np.conj(local_inc_coefs)))
    W_ex = sigma_ex * prefactor
    sigma_ex = W_ex / ps.intensity_incident_field
    return sigma_ex


def cross_section(solution_coefficients, ps, order):
    r"""Counts scattering and extinction cross sections"""
    sigma_sc = scattering_cs(solution_coefficients, ps, order)
    sigma_ex = extinction_cs(solution_coefficients, ps)
    return sigma_sc, sigma_ex
