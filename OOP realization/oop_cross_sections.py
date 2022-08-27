import math
import wavefunctions as wvfs
import numpy as np


def scattering_cs(ps, order):
    r"""Counts scattering cross section"""
    sigma_sc1 = np.zeros(len(ps.spheres))
    sigma_sc2 = np.zeros((ps.num_sph * (order + 1) ** 2) ** 2)
    idx2 = 0
    for sph in range(ps.num_sph):
        scattered_coefs_sph = ps.spheres[sph].scattered_field.coefficients
        sigma_sc1[sph] = math.fsum(np.abs(scattered_coefs_sph) ** 2)
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for osph in np.where(np.arange(ps.num_sph) != sph)[0]:
                scattered_coefs_osph = ps.spheres[osph].scattered_field.coefficients
                for mu, nu in wvfs.multipoles(order):
                    imunu = nu ** 2 + nu + mu
                    distance = ps.spheres[sph].pos - ps.spheres[osph].pos
                    sigma_sc2[idx2] = np.real(np.conj(scattered_coefs_sph[imn]) * scattered_coefs_osph[imunu] * \
                                      wvfs.regular_separation_coefficient(mu, m, nu, n, ps.k_fluid, distance))
                    idx2 += 1
    dimensional_coef = ps.incident_field.ampl ** 2 / (2 * ps.incident_field.omega * ps.fluid.rho * ps.k_fluid)
    sigma_sc = (math.fsum(sigma_sc1) + math.fsum(sigma_sc2)) * dimensional_coef / ps.intensity_incident_field
    return sigma_sc


def extinction_cs(ps):
    r"""Counts extinction cross section"""
    sigma_ex_array = np.zeros(len(ps.spheres))
    for s, particle in enumerate(ps.spheres):
        scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
        if ps.interface:
            incident_coefs += particle.reflected_field.coefficients
        sigma_ex_array[s] = math.fsum(np.real(scattered_coefs * np.conj(incident_coefs)))
    dimensional_coef = ps.incident_field.ampl ** 2 / (2 * ps.incident_field.omega * ps.fluid.rho * ps.k_fluid)
    sigma_ex = -math.fsum(sigma_ex_array) * dimensional_coef / ps.intensity_incident_field
    return sigma_ex


def cross_section(ps, order):
    r"""Counts scattering and extinction cross sections"""
    sigma_sc = scattering_cs(ps, order)
    sigma_ex = extinction_cs(ps)
    return sigma_sc, sigma_ex
