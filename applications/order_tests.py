import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system import LinearSystem
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.fields_expansions import StandingWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def four_silica_aerogel_sphere_in_standing_wave_ls(order, n_sph):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    freq = 12000  # [Hz]
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    r_sph = 0.01  # [m]
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)

    if n_sph == 1:
        pos1 = np.array([0, 0, 0])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1])
    if n_sph == 2:
        pos1, pos2 = np.array([0.015, 0, 0]), np.array([-0.015, 0, 0])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2])
    if n_sph == 3:
        pos1, pos2, pos3 = np.array([-0.015, 0, -0.0145]), np.array([0.015, 0, -0.0145]), np.array([0, 0, 0.0145])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3])
    if n_sph == 4:
        pos1, pos2 = np.array([0, 0, 0.013]), np.array([0, 0, -0.013])
        pos3, pos4 = np.array([0.013, 0, 0]), np.array([-0.013, 0, 0])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere4 = Particle(pos4, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3, sphere4])

    ls = LinearSystem(particles, None, fluid, freq, order)
    return ls


def sub_proc(s, param, freq_result):
    r"""Counts scattering and extinction cross section for physical system with given frequency"""
    lin_sys = four_silica_aerogel_sphere_in_standing_wave_ls(s, param)
    lin_sys.solve()
    ecs = cs.extinction_cs(lin_sys.particles, lin_sys.medium, 12000)
    all_forces = forces.all_forces(lin_sys.particles, lin_sys.medium)
    freq_result.put((ecs, np.concatenate(all_forces)))


def main_proc(sample, param, tab_len):
    r"""Counts scattering cross sections for all frequencies"""
    table = np.zeros((len(sample), tab_len), dtype=float)
    for i, s in enumerate(sample):
        print("Order:", i, "of", len(sample) - 1)
        queue_res = Queue()
        sub_process = Process(target=sub_proc, args=(s, param, queue_res,))
        sub_process.start()
        ecs, frcs = queue_res.get()
        table[i, 0] = ecs
        table[i, 1:] = frcs
        sub_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\ordersph_csv\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    orders = np.arange(4, 13)

    header = ["ord"]+["ecs", "f1x", "f1y", "f1z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]
    tot_table[:, 1:] = main_proc(orders, 1, len(header) - 1)
    write_csv(tot_table, header, "order1sph")

    header = ["ord"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]
    tot_table[:, 1:] = main_proc(orders, 2, len(header) - 1)
    write_csv(tot_table, header, "order2sph")

    header = ["ord"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]
    tot_table[:, 1:] = main_proc(orders, 3, len(header) - 1)
    write_csv(tot_table, header, "order3sph")

    header = ["ord"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z", "f4x", "f4y", "f4z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]
    tot_table[:, 1:] = main_proc(orders, 4, len(header) - 1)
    write_csv(tot_table, header, "order4sph")
