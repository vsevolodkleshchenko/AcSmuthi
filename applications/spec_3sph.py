import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system import LinearSystem
from acsmuthi.postprocessing import forces
import math
from acsmuthi.utility import mathematics as mths

from acsmuthi.fields_expansions import StandingWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def mtpl_extinction_cs(particles, medium, freq, layer=None):
    block_size = len(particles[0].incident_field.coefficients)
    sigma_ex_array = np.zeros((len(particles), block_size), dtype=complex)
    for s, particle in enumerate(particles):
        scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
        if layer:
            incident_coefs += particle.reflected_field.coefficients
        sigma_ex_array[s] = scattered_coefs * np.conj(incident_coefs)
    omega = 2 * np.pi * freq
    dimensional_coef = medium.incident_field.ampl ** 2 / (2 * omega * medium.rho * medium.incident_field.k_l)
    norm = medium.incident_field.intensity(medium.rho, medium.speed_l) * (np.pi * particles[0].r**2)
    ex_poles = -mths.spheres_fsum(sigma_ex_array, block_size) * dimensional_coef / norm
    mono = ex_poles[0]
    return [math.fsum(np.real(ex_poles)), mono]


def three_silica_aerogel_sphere_in_standing_wave_ls(freq):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)

    lda = c_fluid / freq
    pos1, pos2 = np.array([-0.56 * lda, 0, 0]), np.array([0.56*lda, 0, 0])
    pos3 = np.array([0., 0., 0.])

    r_sph = 0.01  # [m]
    r_sph_b = 0.0103
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    order = 12

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere3 = Particle(pos3, r_sph_b, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1, sphere2, sphere3])

    ls = LinearSystem(particles, None, fluid, freq, order)
    return ls


def freq_proc(freq, freq_result):
    r"""Counts scattering and extinction cross section for physical system with given frequency"""
    lin_sys = three_silica_aerogel_sphere_in_standing_wave_ls(freq)
    lin_sys.solve()
    ecs = mtpl_extinction_cs(lin_sys.particles, lin_sys.medium, freq)
    t = np.abs(lin_sys.t_matrix)
    t0 = t[0, 0]
    freq_result.put((ecs, t0))


def spectrum_freq(freqs):
    r"""Counts scattering cross sections for all frequencies"""
    table = np.zeros((len(freqs), 3), dtype=complex)
    for i, freq in enumerate(freqs):
        print("Frequency:", i, "of", len(freqs))
        queue_res = Queue()
        freq_process = Process(target=freq_proc, args=(freq, queue_res,))
        freq_process.start()
        ecss, t_comp = queue_res.get()
        table[i, :2] = ecss[:]
        table[i, 2] = t_comp
        freq_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\final_exp\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    frequencies = np.linspace(2069, 1576, 90, dtype=float)
    spectrum_table = np.zeros((len(frequencies), 4), dtype=complex)
    spectrum_table[:, 0] = 0.02 * frequencies / 331
    spectrum_table[:, 1:] = spectrum_freq(frequencies)
    header = ["dl"]+["cs", "cs0", "t0"]
    write_csv(spectrum_table, header, "spectrum_mltplecs_forces_3aerogelSW12")