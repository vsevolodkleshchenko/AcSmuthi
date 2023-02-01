import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system import LinearSystem
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.fields_expansions import PlaneWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def two_fluid_spheres_ls(pos1, pos2, freq):
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([-0.70711, 0, 0.70711])
    p0 = 100  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 8

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1, sphere2])

    ls = LinearSystem(particles, None, fluid, freq, order)
    return ls


def freq_proc(freq, pos1, pos2, freq_result):
    r"""Counts scattering and extinction cross section for physical system with given frequency"""
    lin_sys = two_fluid_spheres_ls(pos1, pos2, freq)
    lin_sys.solve()
    ecs = cs.extinction_cs(lin_sys.particles, lin_sys.medium, freq)
    all_forces = forces.all_forces(lin_sys.particles, lin_sys.medium)
    freq_result.put((ecs, np.concatenate(all_forces)))


def spectrum_freq_proc(freqs, pos1, pos2, spec_result):
    r"""Counts scattering cross sections for all frequencies"""
    ecs = np.zeros(len(freqs), dtype=float)
    frcs = np.zeros((len(freqs), 6), dtype=float)
    for i, freq in enumerate(freqs):
        print("     Frequency:", i)
        queue_res = Queue()
        freq_process = Process(target=freq_proc, args=(freq, pos1, pos2, queue_res,))
        freq_process.start()
        ecs[i], frcs[i] = queue_res.get()
        freq_process.join()
    spec_result.put((ecs, frcs))


def spectrum_dist_freq(coord_z, freqs):
    r"""Counts scattering cross sections for all distances between spheres"""
    table = np.zeros((len(freqs), 7 * len(coord_z)))
    for i, z in enumerate(coord_z):
        print("Position:", i)
        pos1 = np.array([0, 0, -z])
        pos2 = np.array([0, 0, z])
        queue_res = Queue()
        dist_process = Process(target=spectrum_freq_proc, args=(freqs, pos1, pos2, queue_res,))
        dist_process.start()
        ecs, frcs = queue_res.get()
        table[:, 7 * i] = ecs[:]
        table[:, 7*i+1:7*i+7] = frcs[:]
        dist_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    frequencies = np.linspace(5, 500, 60, dtype=float)
    coord_z = np.array([1.25, 2., 3])
    spectrum_table = np.zeros((len(frequencies), 7 * len(coord_z) + 1))
    spectrum_table[:, 0] = 2 * np.pi * frequencies / 331
    spectrum_table[:, 1:] = spectrum_dist_freq(coord_z, frequencies)
    header = ["ka"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z"] * len(coord_z)
    write_csv(spectrum_table, header, "spectrum_dist_ecs_forces_angle_big")
