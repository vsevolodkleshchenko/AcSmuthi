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
    ro_fluid = 1.225
    c_fluid = 331
    direction = np.array([0, 0, 1])
    freq = 14000
    p0 = 10000
    k_l = 2 * np.pi * freq / c_fluid
    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    r_sph = 0.01
    ro_sph = 80
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))
    c_sph_t = np.sqrt(g / ro_sph)
    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(ro_fluid, c_fluid)

    poses = []

    if n_sph == 1:
        poses = [np.array([0, 0, 0])]

    elif n_sph == 2:
        poses = [np.array([-0.02, 0, 0]), np.array([0.02, 0, 0])]

    elif n_sph == 3:
        poses = [np.array([-0.018, 0, -0.018]), np.array([0.018, 0, -0.018]), np.array([0, 0, 0.018])]

    elif n_sph == 4:
        poses = [np.array([-0.018, 0, -0.018]), np.array([0.018, 0, -0.018]),
                 np.array([-0.018, 0, 0.018]), np.array([0.018, 0, 0.018])]

    elif n_sph == 5:
        poses = [np.array([-0.018, 0, -0.018]), np.array([0.018, 0, -0.018]),
                 np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018])]

    elif n_sph == 6:
        poses = [np.array([-0.036, 0, -0.018]), np.array([0, 0, -0.018]), np.array([0.036, 0, -0.018]),
                 np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018])]

    elif n_sph == 7:
        poses = [np.array([-0.036, 0, -0.018]), np.array([0, 0, -0.018]), np.array([0.036, 0, -0.018]),
                 np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018]),
                 np.array([0, 0, -0.05])]

    elif n_sph == 8:
        poses = [np.array([-0.036, 0, -0.018]), np.array([0, 0, -0.018]), np.array([0.036, 0, -0.018]),
                 np.array([-0.036, 0, 0.018]), np.array([0, 0, 0.018]), np.array([0.036, 0, 0.018]),
                 np.array([0, 0, -0.05]), np.array([0, 0, 0.05])]

    else:
        n = int(np.round(np.sqrt(n_sph), 0))
        for i in range(n):
            for j in range(n):
                poses.append(np.array([i - (n - 1) / 2, 0, j - (n - 1) / 2]) * 0.038)

    particles_lst = []
    for pos in poses:
        particles_lst.append(Particle(pos, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t))

    ls = LinearSystem(np.array(particles_lst), fluid, incident_field, freq, order, True)
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


def write_header(n):
    forces_header = []
    for s in range(1, n + 1):
        for ax in ("x", "y", "z"):
            forces_header.append(f"f{s}"+ax)
    return ["order", "ecs"] + forces_header + ["t_s", "t_cs", "t_f"]


if __name__ == '__main__':
    # orders = np.arange(4, 19)
    # for n_s in range(1, 9):
    #     header = write_header(n_s)
    #     tot_table = np.zeros((len(orders), len(header)))
    #     tot_table[:, 0] = orders[:]
    #     tot_table[:, 1:] = main_proc(orders, n_s, len(header) - 1)
    #     write_csv(tot_table, header, f"{n_s}_sph_order")

    orders = np.arange(3, 11)
    for n_s in np.arange(3, 8) ** 2:
        header = write_header(n_s)
        tot_table = np.zeros((len(orders), len(header)))
        tot_table[:, 0] = orders[:]
        tot_table[:, 1:] = main_proc(orders, n_s, len(header) - 1)
        write_csv(tot_table, header, f"{n_s}_sph_order")
