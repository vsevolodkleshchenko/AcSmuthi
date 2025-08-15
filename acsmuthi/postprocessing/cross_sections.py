import math
import numpy as np

from acsmuthi.simulation import Simulation
from acsmuthi.utility import wavefunctions as wvfs

# todo: this module doesn't work properly for substrate; read and decide what to do


def scattering_cs(simulation: Simulation):
    particles = simulation.particles
    medium = simulation.medium
    initial_field = simulation.initial_field

    freq = initial_field.freq
    k = medium.sur_medium.wavenumber(freq)
    omega = 2 * np.pi * freq

    sigma_sc1 = np.zeros(len(particles))
    sigma_sc2 = np.zeros(simulation.linear_system.t_matrix.shape[0] ** 2)

    idx2 = 0
    for i_p in range(len(particles)):
        scat_cfs_ip = particles[i_p].scattered_field.coefficients
        sigma_sc1[i_p] = math.fsum(np.abs(scat_cfs_ip) ** 2)

        for m, n in wvfs.mn_idx(particles[i_p].n_max):
            imn = n ** 2 + n + m

            for j_p in np.where(np.arange(len(particles)) != i_p)[0]:
                scat_cfs_jp = particles[j_p].scattered_field.coefficients

                for mu, nu in wvfs.mn_idx(particles[i_p].n_max):
                    imunu = nu ** 2 + nu + mu
                    distance = particles[i_p].position - particles[j_p].position
                    sep_coef = wvfs.regular_separation_coefficient(mu, m, nu, n, k, distance)
                    scat_mul = np.conj(scat_cfs_ip[imn]) * scat_cfs_jp[imunu]
                    sigma_sc2[idx2] = np.real(scat_mul * sep_coef)

                    idx2 += 1

    sigma_sc_sum = math.fsum(sigma_sc1) + math.fsum(sigma_sc2)

    coef = initial_field.amplitude ** 2 / (2 * omega * medium.sur_medium.density * k)
    geoms_cs = (np.pi * particles[0].radius ** 2)
    sigma_sc = sigma_sc_sum * coef / initial_field.intensity(medium)
    return sigma_sc / geoms_cs


def extinction_cs(simulation: Simulation, by_multipoles=False):
    particles = simulation.particles
    medium = simulation.medium
    initial_field = simulation.initial_field

    freq = initial_field.freq
    omega = 2 * np.pi * freq
    k = medium.sur_medium.wavenumber(freq)
    coef = initial_field.amplitude ** 2 / (2 * omega * medium.sur_medium.density * k)

    if by_multipoles:
        block_size = len(particles[0].incident_field.coefficients)
        order = int(np.sqrt(block_size) - 1)
        extinction_array = np.zeros((len(particles), block_size))

        for s, particle in enumerate(particles):
            scattered_coefs = particle.scattered_field.coefficients
            incident_coefs = particle.incident_field.coefficients
            extinction_array[s] = np.real(scattered_coefs * np.conj(incident_coefs))

        extinction_poles_array = -np.sum(extinction_array, axis=0)
        extinction_poles = [extinction_poles_array[0]]

        for n in range(1, order + 1):
            extinction_poles.append(extinction_poles_array[n ** 2:(n + 1) ** 2].sum())
        extinction = np.array(extinction_poles)

    else:
        extinction_array = np.zeros(len(particles))

        for s, particle in enumerate(particles):
            scattered_coefs = particle.scattered_field.coefficients
            incident_coefs = particle.incident_field.coefficients
            extinction_array[s] = math.fsum(np.real(scattered_coefs * np.conj(incident_coefs)))
        extinction = -np.sum(extinction_array)

    return extinction * coef / initial_field.intensity(medium)


def cross_section(simulation):
    sigma_sc = scattering_cs(simulation)
    sigma_ex = extinction_cs(simulation)
    return sigma_sc, sigma_ex
