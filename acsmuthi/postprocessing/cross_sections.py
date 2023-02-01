import math
from acsmuthi.utility import wavefunctions as wvfs
import numpy as np


def scattering_cs(particles, medium, freq, order):
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
                                              wvfs.regular_separation_coefficient(mu, m, nu, n, medium.incident_field.k_l, distance))
                    idx2 += 1
    omega = 2*np.pi*freq
    dimensional_coef = medium.incident_field.ampl ** 2 / (2 * omega * medium.rho * medium.incident_field.k_l)
    sigma_sc = (math.fsum(sigma_sc1) + math.fsum(sigma_sc2)) * dimensional_coef / medium.incident_field.intensity(medium.rho, medium.speed_l)
    return sigma_sc / (np.pi * particles[0].r**2)


def extinction_cs(particles, medium, freq, layer=None):
    r"""Counts extinction cross section"""
    sigma_ex_array = np.zeros(len(particles))
    for s, particle in enumerate(particles):
        scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
        if layer:
            incident_coefs += particle.reflected_field.coefficients
        sigma_ex_array[s] = math.fsum(np.real(scattered_coefs * np.conj(incident_coefs)))
    omega = 2*np.pi*freq
    dimensional_coef = medium.incident_field.ampl ** 2 / (2 * omega * medium.rho * medium.incident_field.k_l)
    sigma_ex = -math.fsum(sigma_ex_array) * dimensional_coef / medium.incident_field.intensity(medium.rho, medium.speed_l)
    return sigma_ex / (np.pi * particles[0].r**2)


def cross_section(particles, medium, freq, order, layer):
    r"""Counts scattering and extinction cross sections"""
    sigma_sc = scattering_cs(particles, medium, freq, order)
    sigma_ex = extinction_cs(particles, medium, freq, layer=layer)
    return sigma_sc, sigma_ex
