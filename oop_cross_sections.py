import math
import wavefunctions as wvfs
import numpy as np


def scattering_cs(incident_field, particles, fluid, freq, order):
    r"""Counts scattering cross section"""
    sigma_sc1 = np.zeros(len(particles))
    sigma_sc2 = np.zeros((len(particles) * (order + 1) ** 2) ** 2)
    idx2 = 0
    for sph in range(len(particles)):
        scattered_coefs_sph = particles[sph].scattered_field.coefficients
        sigma_sc1[sph] = math.fsum(np.abs(scattered_coefs_sph) ** 2)
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for osph in np.where(np.arange(len(particles)) != sph)[0]:
                scattered_coefs_osph = particles[osph].scattered_field.coefficients
                for mu, nu in wvfs.multipoles(order):
                    imunu = nu ** 2 + nu + mu
                    distance = particles[sph].pos - particles[osph].pos
                    sigma_sc2[idx2] = np.real(np.conj(scattered_coefs_sph[imn]) * scattered_coefs_osph[imunu] * \
                                      wvfs.regular_separation_coefficient(mu, m, nu, n, incident_field.k, distance))
                    idx2 += 1
    omega = 2*np.pi*freq
    dimensional_coef = incident_field.ampl ** 2 / (2 * omega * fluid.rho * incident_field.k)
    sigma_sc = (math.fsum(sigma_sc1) + math.fsum(sigma_sc2)) * dimensional_coef / incident_field.intensity(fluid.rho, fluid.speed)
    return sigma_sc


def extinction_cs(incident_field, particles, fluid, freq, layer=None):
    r"""Counts extinction cross section"""
    sigma_ex_array = np.zeros(len(particles))
    for s, particle in enumerate(particles):
        scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
        if layer:
            incident_coefs += particle.reflected_field.coefficients
        sigma_ex_array[s] = math.fsum(np.real(scattered_coefs * np.conj(incident_coefs)))
    omega = 2*np.pi*freq
    dimensional_coef = incident_field.ampl ** 2 / (2 * omega * fluid.rho * incident_field.k)
    sigma_ex = -math.fsum(sigma_ex_array) * dimensional_coef / incident_field.intensity(fluid.rho, fluid.speed)
    return sigma_ex


def cross_section(incident_field, particles, fluid, freq, order, layer):
    r"""Counts scattering and extinction cross sections"""
    sigma_sc = scattering_cs(incident_field, particles, fluid, freq, order)
    sigma_ex = extinction_cs(incident_field, particles, fluid, freq, layer=layer)
    return sigma_sc, sigma_ex
