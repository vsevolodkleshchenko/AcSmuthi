import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system import LinearSystem
from acsmuthi.postprocessing import forces
from acsmuthi.postprocessing import cross_sections as cs

from acsmuthi.fields_expansions import StandingWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def three_silica_aerogel_sphere_in_standing_wave_ls(distance, freq):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)

    pos1, pos2 = np.array([-distance, 0, 0]), np.array([distance, 0, 0])
    pos3 = np.array([0., 0., 0.])

    r_sph = 0.01  # [m]
    r_sph_b = 0.0103
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    order = 1

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere3 = Particle(pos3, r_sph_b, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1, sphere2, sphere3])

    ls = LinearSystem(particles, None, fluid, freq, order)
    return ls


def dist_proc(dist, freq, freq_result):
    r"""Counts scattering and extinction cross section for physical system with given frequency"""
    lin_sys = three_silica_aerogel_sphere_in_standing_wave_ls(dist, freq)
    lin_sys.solve()
    ecs = cs.extinction_cs(lin_sys.particles, lin_sys.medium, freq)
    all_forces = forces.all_forces(lin_sys.particles, lin_sys.medium)
    freq_result.put((ecs, np.concatenate(all_forces)))


def forces_dist(dists, freq):
    r"""Counts scattering cross sections for all frequencies"""
    table = np.zeros((len(dists), 10), dtype=float)
    for i, dist in enumerate(dists):
        print("Distance:", i, "of", len(dists))
        queue_res = Queue()
        dist_process = Process(target=dist_proc, args=(dist, freq, queue_res,))
        dist_process.start()
        ecs, frcs = queue_res.get()
        table[i, 0] = ecs
        table[i, 1:] = frcs
        dist_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\final_exp\\dream1\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    header = ["dl"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z"]

    # fr = 331 / 0.02 / 8.99
    # d_max = 3 * 331 / fr
    # distances = np.linspace(0.024, d_max / 2, 80, dtype=float)
    # dist_table = np.zeros((len(distances), 11))
    # dist_table[:, 0] = fr * distances / 331
    # dist_table[:, 1:] = forces_dist(distances, fr)
    # write_csv(dist_table, header, "forces_3aerogelSW_l_mo")
    #
    # fr = 331 / 0.02 / 9.27
    # d_max = 3 * 331 / fr
    # distances = np.linspace(0.024, d_max / 2, 80, dtype=float)
    # dist_table = np.zeros((len(distances), 11))
    # dist_table[:, 0] = fr * distances / 331
    # dist_table[:, 1:] = forces_dist(distances, fr)
    # write_csv(dist_table, header, "forces_3aerogelSW_m_mo")
    #
    # fr = 331 / 0.02 / 9.55
    # d_max = 3 * 331 / fr
    # distances = np.linspace(0.024, d_max, 90, dtype=float)
    # dist_table = np.zeros((len(distances), 11))
    # dist_table[:, 0] = fr * distances / 331
    # dist_table[:, 1:] = forces_dist(distances, fr)
    # write_csv(dist_table, header, "forces_3aerogelSW_r")

    # dist_table = np.zeros((80, 11))
    # lD = np.linspace(8.9, 12, 30)
    # for lDi in lD:
    #     fr = 331 / 0.02 / lDi
    #     d_max = 3 * 331 / fr
    #     distances = np.linspace(0.024, d_max / 2, 80, dtype=float)
    #     dist_table[:, 0] = fr * distances / 331
    #     dist_table[:, 1:] = forces_dist(distances, fr)
    #     write_csv(dist_table, header, f"forces_3aerogelSW_{lDi}_mo")

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]
    print(c_sph_t, c_sph_l)
