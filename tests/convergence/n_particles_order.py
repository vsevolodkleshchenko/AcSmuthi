import time

import numpy as np
import csv

from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def silica_aerogel_spheres_in_plane_wave_ls(order, n_sph):
    ro_fluid = 1.225
    c_fluid = 331
    direction = np.array([0.70711, 0, 0.70711])
    freq = 140  # for D/l ~ 0.85; 14 for D/l ~ 0.1
    p0 = 1
    k_l = 2 * np.pi * freq / c_fluid
    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    r_sph = 1
    ro_sph = 80
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))
    c_sph_t = np.sqrt(g / ro_sph)
    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(ro_fluid, c_fluid)

    d_har = 1.8 * r_sph  # for freq=140 ; 0.4 * c_fluid / freq  # for freq=14
    poses = []

    if n_sph == 1:
        poses = [np.array([0., 0., 0.])]

    elif n_sph == 2:
        poses = [np.array([-1.1 * d_har, 0., 0.]), np.array([1.1 * d_har, 0., 0.])]

    elif n_sph == 3:
        poses = [np.array([-d_har, 0, -d_har]), np.array([d_har, 0, -d_har]), np.array([0, 0, d_har])]

    elif n_sph == 4:
        poses = [np.array([-d_har, 0, -d_har]), np.array([d_har, 0, -d_har]),
                 np.array([-d_har, 0, d_har]), np.array([d_har, 0, d_har])]

    elif n_sph == 5:
        poses = [np.array([-d_har, 0, -d_har]), np.array([d_har, 0, -d_har]),
                 np.array([-2*d_har, 0, d_har]), np.array([0, 0, d_har]), np.array([2*d_har, 0, d_har])]

    elif n_sph == 6:
        poses = [np.array([-2*d_har, 0, -d_har]), np.array([0, 0, -d_har]), np.array([2*d_har, 0, -d_har]),
                 np.array([-2*d_har, 0, d_har]), np.array([0, 0, d_har]), np.array([2*d_har, 0, d_har])]

    elif n_sph == 7:
        poses = [np.array([-2*d_har, 0, -d_har]), np.array([0, 0, -d_har]), np.array([2*d_har, 0, -d_har]),
                 np.array([-2*d_har, 0, d_har]), np.array([0, 0, d_har]), np.array([2*d_har, 0, d_har]),
                 np.array([0, 0, -2.8*d_har])]

    elif n_sph == 8:
        poses = [np.array([-2*d_har, 0, -d_har]), np.array([0, 0, -d_har]), np.array([2*d_har, 0, -d_har]),
                 np.array([-2*d_har, 0, d_har]), np.array([0, 0, d_har]), np.array([2*d_har, 0, d_har]),
                 np.array([0, 0, -2.8*d_har]), np.array([0, 0, 2.8*d_har])]

    else:
        n = int(np.round(np.sqrt(n_sph), 0))
        d_h = 3.8 * r_sph  # for freq=140 ; 0.7 * c_fluid / freq for freq=14
        for i in range(n):
            for j in range(n):
                poses.append(np.array([i - (n - 1) / 2, 0, j - (n - 1) / 2]) * d_h)

    particles_lst = []
    for pos in poses:
        particles_lst.append(Particle(pos, r_sph, ro_sph, c_sph_l, order))
    particles = np.array(particles_lst)

    sim = Simulation(particles, fluid, incident_field, freq, order, True)
    t_prep, t_sol = sim.run()
    t_so = time.time()
    ecs = cs.extinction_cs(sim.particles, sim.medium, sim.incident_field, freq)
    t_cs = time.time()
    all_forces = forces.all_forces(sim.particles, sim.medium, sim.incident_field)
    t_fr = time.time()
    return ecs, np.concatenate(all_forces), [t_prep+t_sol, t_cs-t_so, t_fr-t_cs]


def main_proc(orders, n_sph):
    tab_len = 1 + 3 * n_sph + 3
    table = np.zeros((len(orders), tab_len), dtype=float)
    for i, order in enumerate(orders):
        print("    Order:", i, "of", len(orders) - 1)
        ecs, frcs, timing = silica_aerogel_spheres_in_plane_wave_ls(order, n_sph)
        table[i, 0] = ecs
        table[i, 1:tab_len-3] = frcs
        table[i, tab_len-3:] = timing
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\n_particles_order_csv\\Dll_1\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def write_header(n):
    forces_header = []
    for s in range(1, n + 1):
        for ax in ("x", "y", "z"):
            forces_header.append(f"f{s}"+ax)
    return ["order", "ecs"] + forces_header + ["t_s", "t_cs", "t_f"]


def main():
    orders = np.arange(1, 13)  # for freq=140; np.arange(1, 7)  for freq=14
    for n_s in range(1, 9):
        header = write_header(n_s)
        tot_table = np.zeros((len(orders), len(header)))
        tot_table[:, 0] = orders[:]
        print(f"Number of particles: {n_s}")
        tot_table[:, 1:] = main_proc(orders, n_s)
        write_csv(tot_table, header, f"{n_s}_sph_order")

    orders = np.arange(1, 10)   # for freq=140; np.arange(1, 7) for freq=14
    for n_s in np.arange(3, 8) ** 2:
        header = write_header(n_s)
        tot_table = np.zeros((len(orders), len(header)))
        tot_table[:, 0] = orders[:]
        print(f"Number of particles: {n_s}")
        tot_table[:, 1:] = main_proc(orders, n_s)
        write_csv(tot_table, header, f"{n_s}_sph_order")


main()
