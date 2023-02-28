import time

import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def silica_aerogel_spheres_in_plane_wave_ls(order, n_sph):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    freq = 14000  # [Hz]
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    r_sph = 0.01  # [m]
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(ro_fluid, c_fluid)

    if n_sph == 1:
        pos1 = np.array([0, 0, 0])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1])
    if n_sph == 2:
        pos1, pos2 = np.array([-0.02, 0, 0]), np.array([0.02, 0, 0])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2])
    if n_sph == 3:
        pos1, pos2, pos3 = np.array([-0.018, 0, -0.018]), np.array([0.018, 0, -0.018]), np.array([0, 0, 0.018])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3])
    if n_sph == 4:
        pos1, pos2 = np.array([-0.018, 0, -0.018]), np.array([0.018, 0, -0.018])
        pos3, pos4 = np.array([-0.018, 0, 0.018]), np.array([0.018, 0, 0.018])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere4 = Particle(pos4, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3, sphere4])
    if n_sph == 5:
        pos1, pos2 = np.array([-0.018, 0, -0.018]), np.array([0.018, 0, -0.018])
        pos3, pos4, pos5 = np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere4 = Particle(pos4, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere5 = Particle(pos5, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3, sphere4, sphere5])
    if n_sph == 6:
        pos1, pos2, pos3 = np.array([-0.036, 0, -0.018]), np.array([0, 0, -0.018]), np.array([0.036, 0, -0.018])
        pos4, pos5, pos6 = np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere4 = Particle(pos4, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere5 = Particle(pos5, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere6 = Particle(pos6, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3, sphere4, sphere5, sphere6])

    if n_sph == 7:
        pos1, pos2, pos3 = np.array([-0.036, 0, -0.018]), np.array([0, 0, -0.018]), np.array([0.036, 0, -0.018])
        pos4, pos5, pos6 = np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018])
        pos7 = np.array([0, 0, -0.036])
        sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere4 = Particle(pos4, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere5 = Particle(pos5, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere6 = Particle(pos6, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        sphere7 = Particle(pos7, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
        particles = np.array([sphere1, sphere2, sphere3, sphere4, sphere5, sphere6, sphere7])

    ls = LinearSystem(particles, fluid, incident_field, freq, order, True)
    return ls


def sub_proc(s, param, freq_result):
    lin_sys = silica_aerogel_spheres_in_plane_wave_ls(s, param)
    t_st = time.time()
    lin_sys.prepare()
    lin_sys.solve()
    t_so = time.time()
    ecs = cs.extinction_cs(lin_sys.particles, lin_sys.medium, lin_sys.incident_field, 12000)
    t_cs = time.time()
    all_forces = forces.all_forces(lin_sys.particles, lin_sys.medium, lin_sys.incident_field)
    t_fr = time.time()
    freq_result.put((ecs, np.concatenate(all_forces), [t_so-t_st, t_cs-t_so, t_fr-t_cs]))


def main_proc(sample, param, tab_len):
    table = np.zeros((len(sample), tab_len), dtype=float)
    for i, s in enumerate(sample):
        print("Order:", i, "of", len(sample) - 1)
        queue_res = Queue()
        sub_process = Process(target=sub_proc, args=(s, param, queue_res,))
        sub_process.start()
        ecs, frcs, timing = queue_res.get()
        table[i, 0] = ecs
        table[i, 1:tab_len-3] = frcs
        table[i, tab_len-3:] = timing
        sub_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\n_sph_order_csv\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    orders = np.arange(4, 19)

    # header = ["order"]+["ecs", "f1x", "f1y", "f1z", "t_s", "t_cs", "t_f"]
    # tot_table = np.zeros((len(orders), len(header)))
    # tot_table[:, 0] = orders[:]
    # tot_table[:, 1:] = main_proc(orders, 1, len(header) - 1)
    # write_csv(tot_table, header, "1_sph_order")
    #
    # header = ["order"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "t_s", "t_cs", "t_f"]
    # tot_table = np.zeros((len(orders), len(header)))
    # tot_table[:, 0] = orders[:]
    # tot_table[:, 1:] = main_proc(orders, 2, len(header) - 1)
    # write_csv(tot_table, header, "2_sph_order")
    #
    # header = ["order"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z", "t_s", "t_cs", "t_f"]
    # tot_table = np.zeros((len(orders), len(header)))
    # tot_table[:, 0] = orders[:]
    # tot_table[:, 1:] = main_proc(orders, 3, len(header) - 1)
    # write_csv(tot_table, header, "3_sph_order")

    # header = ["order"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z", "f4x",
    #                     "f4y", "f4z", "t_s", "t_cs", "t_f"]
    # tot_table = np.zeros((len(orders), len(header)))
    # tot_table[:, 0] = orders[:]
    # tot_table[:, 1:] = main_proc(orders, 4, len(header) - 1)
    # write_csv(tot_table, header, "4_sph_order")

    # header = ["order"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z",
    #                     "f4x", "f4y", "f4z", "f5x", "f5y", "f5z", "t_s", "t_cs", "t_f"]
    # tot_table = np.zeros((len(orders), len(header)))
    # tot_table[:, 0] = orders[:]
    # tot_table[:, 1:] = main_proc(orders, 5, len(header) - 1)
    # write_csv(tot_table, header, "5_sph_order")

    header = ["order"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z",
                        "f4x", "f4y", "f4z", "f5x", "f5y", "f5z", "f6x", "f6y", "f6z", "t_s", "t_cs", "t_f"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]
    tot_table[:, 1:] = main_proc(orders, 6, len(header) - 1)
    write_csv(tot_table, header, "6_sph_order")

    header = ["order"]+["ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z", "f3x", "f3y", "f3z",
                        "f4x", "f4y", "f4z", "f5x", "f5y", "f5z", "f6x", "f6y", "f6z",
                        "f7x", "f7y", "f7z", "t_s", "t_cs", "t_f"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]
    tot_table[:, 1:] = main_proc(orders, 7, len(header) - 1)
    write_csv(tot_table, header, "7_sph_order")
