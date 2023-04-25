import time

import numpy as np
import csv

from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium


'''
Test to check the convergence of scattering cross sections and acoustic forces for a different number of particles 
that are located at a distance of the order of wavelength, depending on the multipole order, 
as well as time measurement
'''

# Parameters: steel spheres in air in plane wave
rho_medium, c_medium = 1.225, 331
rho_sph, cp_sph, cs_sph = 7700, 5740, 3092
r_sph = 1
p0, direction = 1, np.array([0.70711, 0, 0.70711])

# Main parameter for changing scattering regime is frequency
freq = 150
k, lda = 2 * np.pi * freq / c_medium, c_medium / freq
char_dist = lda


def create_positions(n_sph):
    poses = []
    if n_sph == 1:
        poses = [np.array([0., 0., 0.])]

    elif n_sph == 2:
        poses = [np.array([-char_dist, 0., 0.]),
                 np.array([char_dist, 0., 0.])]

    elif n_sph == 3:
        poses = [np.array([-char_dist, 0, -char_dist]),
                 np.array([char_dist, 0, -char_dist]),
                 np.array([0, 0, char_dist])]

    elif n_sph == 4:
        poses = [np.array([-char_dist, 0, -char_dist]),
                 np.array([char_dist, 0, -char_dist]),
                 np.array([-char_dist, 0, char_dist]),
                 np.array([char_dist, 0, char_dist])]

    elif n_sph == 5:
        poses = [np.array([-char_dist, 0, -char_dist]),
                 np.array([char_dist, 0, -char_dist]),
                 np.array([-2 * char_dist, 0, char_dist]),
                 np.array([0, 0, char_dist]),
                 np.array([2 * char_dist, 0, char_dist])]

    elif n_sph == 6:
        poses = [np.array([-2 * char_dist, 0, -char_dist]),
                 np.array([0, 0, -char_dist]),
                 np.array([2 * char_dist, 0, -char_dist]),
                 np.array([-2 * char_dist, 0, char_dist]),
                 np.array([0, 0, char_dist]),
                 np.array([2 * char_dist, 0, char_dist])]

    elif n_sph == 7:
        poses = [np.array([-2 * char_dist, 0, -char_dist]),
                 np.array([0, 0, -char_dist]),
                 np.array([2 * char_dist, 0, -char_dist]),
                 np.array([-2 * char_dist, 0, char_dist]),
                 np.array([0, 0, char_dist]),
                 np.array([2 * char_dist, 0, char_dist]),
                 np.array([0, 0, -3 * char_dist])]

    elif n_sph == 8:
        poses = [np.array([-2 * char_dist, 0, -char_dist]),
                 np.array([0, 0, -char_dist]),
                 np.array([2 * char_dist, 0, -char_dist]),
                 np.array([-2 * char_dist, 0, char_dist]),
                 np.array([0, 0, char_dist]),
                 np.array([2 * char_dist, 0, char_dist]),
                 np.array([0, 0, -3 * char_dist]),
                 np.array([0, 0, 3 * char_dist])]

    else:
        n = int(np.round(np.sqrt(n_sph), 0))
        for i in range(n):
            for j in range(n):
                poses.append(np.array([i - (n - 1) / 2, 0, j - (n - 1) / 2]) * 2 * char_dist)
    return poses


def main_proc(orders, n_sph):
    tab_len = 1 + 3 * n_sph + 3
    table = np.zeros((len(orders), tab_len), dtype=float)
    positions = create_positions(n_sph)
    for i, order in enumerate(orders):
        print("    Order:", i, "of", len(orders) - 1)

        initial_field = PlaneWave(k, p0, direction)
        medium = Medium(rho_medium, c_medium)
        particles_lst = []
        for pos in positions:
            particles_lst.append(SphericalParticle(pos, r_sph, rho_sph, cp_sph, order, cs_sph))
        particles = np.array(particles_lst)

        sim = Simulation(particles, medium, initial_field, freq, order, True)
        t_prep, t_sol = sim.run()
        t_so = time.time()
        ecs = cs.extinction_cs(sim.particles, sim.medium, sim.initial_field, freq)
        t_cs = time.time()
        all_forces = forces.all_forces(sim.particles, sim.medium, sim.initial_field)
        t_fr = time.time()

        table[i, 0] = ecs
        table[i, 1:tab_len-3] = np.concatenate(all_forces)
        table[i, tab_len-3:] = [t_prep+t_sol, t_cs-t_so, t_fr-t_cs]
    return table


def write_csv(data, fieldnames, filename):
    with open(f".\\n_particles_order_csv\\freq{freq}\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
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
    orders = np.arange(1, 10)
    for n_s in range(1, 9):
        header = write_header(n_s)
        tot_table = np.zeros((len(orders), len(header)))
        tot_table[:, 0] = orders[:]
        print(f"Number of particles: {n_s}")
        tot_table[:, 1:] = main_proc(orders, n_s)
        write_csv(tot_table, header, f"{n_s}sph")

    orders = np.arange(1, 10)
    for n_s in np.arange(3, 8) ** 2:
        header = write_header(n_s)
        tot_table = np.zeros((len(orders), len(header)))
        tot_table[:, 0] = orders[:]
        print(f"Number of particles: {n_s}")
        tot_table[:, 1:] = main_proc(orders, n_s)
        write_csv(tot_table, header, f"{n_s}sph")


# main()
