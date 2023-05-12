import math
from acsmuthi.utility import wavefunctions as wvfs
import numpy as np


def scattering_cs(simulation):
    r"""Counts scattering cross section"""
    particles, medium, initial_field = simulation.particles, simulation.medium, simulation.initial_field
    freq, order = simulation.freq, simulation.order
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
                    distance = particles[sph].position - particles[osph].position
                    sigma_sc2[idx2] = np.real(np.conj(scattered_coefs_sph[imn]) * scattered_coefs_osph[imunu] * \
                                              wvfs.regular_separation_coefficient(mu, m, nu, n, initial_field.k, distance))
                    idx2 += 1
    omega = 2*np.pi*freq
    dimensional_coef = initial_field.amplitude ** 2 / (2 * omega * medium.density * initial_field.k)
    sigma_sc = (math.fsum(sigma_sc1) + math.fsum(sigma_sc2)) * dimensional_coef / initial_field.intensity(medium.density, medium.cp)
    return sigma_sc / (np.pi * particles[0].radius ** 2)


def extinction_cs(simulation, by_multipoles=False):
    r"""Counts extinction cross section"""
    particles, medium, initial_field, freq = simulation.particles, simulation.medium, simulation.initial_field, simulation.freq
    omega = 2*np.pi*freq
    dimensional_coef = initial_field.amplitude ** 2 / (2 * omega * medium.density * initial_field.k)

    if by_multipoles:
        block_size = len(particles[0].incident_field.coefficients)
        order = int(np.sqrt(block_size) - 1)
        extinction_array = np.zeros((len(particles), block_size))
        for s, particle in enumerate(particles):
            scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
            extinction_array[s] = np.real(scattered_coefs * np.conj(incident_coefs))
        extinction_poles_array = -np.sum(extinction_array, axis=0)
        extinction_poles = [extinction_poles_array[0]]
        for n in range(1, order + 1):
            extinction_poles.append(np.sum(extinction_poles_array[n ** 2:(n + 1) ** 2]))
        extinction = np.array(extinction_poles)

    else:
        extinction_array = np.zeros(len(particles))
        for s, particle in enumerate(particles):
            scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
            extinction_array[s] = math.fsum(np.real(scattered_coefs * np.conj(incident_coefs)))
        extinction = -np.sum(extinction_array)
    return extinction * dimensional_coef / initial_field.intensity(medium.density, medium.cp)


def cross_section(simulation):
    r"""Counts scattering and extinction cross sections"""
    sigma_sc = scattering_cs(simulation)
    sigma_ex = extinction_cs(simulation)
    return sigma_sc, sigma_ex
