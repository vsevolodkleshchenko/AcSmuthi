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
    r"""Counts extinction cross section"""
    block_size = len(particles[0].incident_field.coefficients)
    sigma_ex_array = np.zeros((len(particles), block_size))
    for s, particle in enumerate(particles):
        scattered_coefs, incident_coefs = particle.scattered_field.coefficients, particle.incident_field.coefficients
        if layer:
            incident_coefs += particle.reflected_field.coefficients
        sigma_ex_array[s] = np.real(scattered_coefs * np.conj(incident_coefs))
    omega = 2 * np.pi * freq
    dimensional_coef = medium.incident_field.ampl ** 2 / (2 * omega * medium.rho * medium.incident_field.k_l)
    norm = medium.incident_field.intensity(medium.rho, medium.speed_l) * (np.pi * particles[0].r**2)
    ex_poles = -np.real(mths.spheres_fsum(sigma_ex_array, block_size) * dimensional_coef / norm)
    mono, dipo, quad = ex_poles[0], math.fsum(ex_poles[1:4]), math.fsum(ex_poles[4:9])
    octu, pole4, pole5 = math.fsum(ex_poles[9:16]), math.fsum(ex_poles[16:25]), math.fsum(ex_poles[25:36])
    return [math.fsum(ex_poles), mono, dipo, quad, octu, pole4, pole5]


def silica_aerogel_sphere_in_standing_wave_ls(freq):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    pos1 = np.array([0, 0, 0])  # [m]
    r_sph = 0.01  # [m]
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    order = 5

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1])

    ls = LinearSystem(particles, None, fluid, freq, order)
    return ls


def freq_proc(freq, freq_result):
    r"""Counts scattering and extinction cross section for physical system with given frequency"""
    lin_sys = silica_aerogel_sphere_in_standing_wave_ls(freq)
    lin_sys.solve()
    ecs = mtpl_extinction_cs(lin_sys.particles, lin_sys.medium, freq)
    t = np.abs(lin_sys.t_matrix)
    t1, t2, t3, t4, t5, t6 = t[0, 0], t[2, 2], t[6, 6], t[12, 12], t[20, 20], t[30, 30]
    freq_result.put((ecs, [t1, t2, t3, t4, t5, t6]))


def spectrum_freq(freqs):
    r"""Counts scattering cross sections for all frequencies"""
    table = np.zeros((len(freqs), 13), dtype=float)
    for i, freq in enumerate(freqs):
        print("Frequency:", i, "of", len(freqs))
        queue_res = Queue()
        freq_process = Process(target=freq_proc, args=(freq, queue_res,))
        freq_process.start()
        ecss, t_comps = queue_res.get()
        table[i, :7] = ecss[:]
        table[i, 7:] = t_comps
        freq_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    frequencies = np.linspace(1000, 20000, 120, dtype=float)
    spectrum_table = np.zeros((len(frequencies), 14))
    spectrum_table[:, 0] = 0.02 * frequencies / 331
    spectrum_table[:, 1:] = spectrum_freq(frequencies)
    header = ["dl"]+["cs", "cs0", "cs1", "cs2", "cs3", "cs4", "cs5", "t0", "t1", "t2", "t3", "t4", "t5"]
    write_csv(spectrum_table, header, "spectrum_mltplecs_forces_1aerogelSW")