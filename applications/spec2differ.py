import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system import LinearSystem
import math
from acsmuthi.utility import mathematics as mths
from acsmuthi.fields_expansions import PlaneWave, StandingWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
import acsmuthi.postprocessing.cross_sections as cs


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
    mono, dipo, quad = ex_poles[0], mths.complex_fsum(ex_poles[1:4]), mths.complex_fsum(ex_poles[4:9])
    octu, pole4, pole5 = mths.complex_fsum(ex_poles[9:16]), mths.complex_fsum(ex_poles[16:25]), mths.complex_fsum(ex_poles[25:36])
    return [math.fsum(np.real(ex_poles)), mono, dipo, quad, octu, pole4, pole5]


def silica_aerogel_sphere_in_plane_wave_ls(freq):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    pos1 = np.array([0, 0, 0])  # [m]
    r_sph = 0.0105  # [m]
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    order = 1

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1])

    ls = LinearSystem(particles, None, fluid, freq, order)
    return ls


def freq_proc(freq, freq_result):
    r"""Counts scattering and extinction cross section for physical system with given frequency"""
    lin_sys = silica_aerogel_sphere_in_plane_wave_ls(freq)
    lin_sys.solve()
    ecs = cs.extinction_cs(lin_sys.particles, lin_sys.medium, freq)
    # ecs = mtpl_extinction_cs(lin_sys.particles, lin_sys.medium, freq)
    t = lin_sys.t_matrix
    t0 = t[0, 0]
    freq_result.put((ecs, t0))


def spectrum_freq(freqs):
    r"""Counts scattering cross sections for all frequencies"""
    table = np.zeros((len(freqs), 2), dtype=complex)
    for i, freq in enumerate(freqs):
        print("Frequency:", i, "of", len(freqs))
        queue_res = Queue()
        freq_process = Process(target=freq_proc, args=(freq, queue_res,))
        freq_process.start()
        ecs, t_0 = queue_res.get()
        table[i, 0] = ecs
        table[i, 1] = t_0
        freq_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\final_exp\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    ld = np.linspace(8., 10.26, 250, dtype=float)
    frequencies = 331 / ld / 0.02
    spectrum_table = np.zeros((len(frequencies), 3), dtype=complex)
    spectrum_table[:, 0] = ld
    spectrum_table[:, 1:] = spectrum_freq(frequencies)
    header = ["ld"]+["cs", "t0"]
    write_csv(spectrum_table, header, "spectrum_mono_aerogelPW_D2.1_f")
