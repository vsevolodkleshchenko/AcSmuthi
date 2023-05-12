import numpy as np
import csv

from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium


'''
Test to check the convergence of scattering cross sections and acoustic forces for a different sizes of 
spherical particle, depending on the multipole order
'''

# Parameters: steel spheres in air in plane wave
rho_medium, c_medium = 1.225, 331
rho_sph, cp_sph, cs_sph = 7700, 5740, 3092
r_sph = 1
p0, direction = 1, np.array([0.70711, 0, 0.70711])

# Main parameter for changing scattering regime is frequency
freq = 5
k, lda = 2 * np.pi * freq / c_medium, c_medium / freq


def main_proc(orders, distance):
    table = np.zeros((len(orders), 7), dtype=float)
    for i, order in enumerate(orders):
        print("     Order:", i, "of", len(orders))

        incident_field = PlaneWave(k, p0, direction)
        medium = Medium(rho_medium, c_medium)

        pos1, pos2 = np.array([distance / 2, 0, 0]), np.array([-distance / 2, 0, 0])
        sphere1 = SphericalParticle(pos1, r_sph, rho_sph, cp_sph, order, cs_sph)
        sphere2 = SphericalParticle(pos2, r_sph, rho_sph, cp_sph, order, cs_sph)
        particles = np.array([sphere1, sphere2])

        sim = Simulation(particles, medium, incident_field, freq, order, True)
        sim.run()
        ecs = cs.extinction_cs(sim)
        all_forces = forces.all_forces_old(particles, medium, incident_field)

        table[i, 0] = ecs
        table[i, 1:] = np.concatenate(all_forces)
    return table


def write_csv(data, fieldnames, filename):
    with open(f".\\distance_order_csv\\freq{freq}\\" + filename + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def main():
    orders = np.arange(1, 11)
    dl = np.array([0.001, 0.01, 0.1, 1, 5, 10, 15, 20])
    distances = 2 * r_sph + dl * lda

    header = ["ord", "ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]

    for i, d in enumerate(distances):
        print(i, "distance of", len(distances)-1)
        tot_table[:, 1:] = main_proc(orders, d)
        write_csv(tot_table, header, f"dl{dl[i]}")


# main()
