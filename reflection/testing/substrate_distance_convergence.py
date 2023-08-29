import numpy as np
import csv

from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium


'''
Test to check the convergence of scattering cross sections and acoustic forces for a different distances from 
the substrate of spherical particle, depending on the multipole order
'''

# Parameters: water spheres in air in plane wave
rho_medium, c_medium = 1.225, 331
rho_sph, cp_sph = 997, 1403
r_sph = 1
p0, direction = 1, np.array([0.70711, 0, -0.70711])

freq = 8
k = 2 * np.pi * freq / c_medium


def main_proc(orders, distance):
    table = np.zeros((len(orders), 4), dtype=float)
    for i, order in enumerate(orders):
        print("     Order:", i, "of", len(orders))

        incident_field = PlaneWave(k, p0, direction)
        medium = Medium(rho_medium, c_medium, is_substrate=True)

        sphere1 = SphericalParticle(np.array([0, 0, distance]), r_sph, rho_sph, cp_sph, order)
        particles = np.array([sphere1])

        sim = Simulation(particles, medium, incident_field, freq, order)
        sim.run()
        ecs = cs.extinction_cs(sim)
        all_forces = forces.all_forces(sim)

        table[i, 0] = ecs
        table[i, 1:] = np.concatenate(all_forces)
    return table


def write_csv(data, fieldnames, filename):
    with open(f".\\subs_distance_order_csv\\freq{freq}\\" + filename + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def main():
    orders = np.arange(1, 11)
    kd = np.array([0.001, 0.01, 0.1, 1, 2, 5, 10, 15])
    distances = r_sph + kd / k

    header = ["ord", "ecs", "fx", "fy", "fz"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]

    for i, d in enumerate(distances):
        print(i, "distance of", len(distances)-1)
        tot_table[:, 1:] = main_proc(orders, d)
        write_csv(tot_table, header, f"kd{kd[i]}")


# main()
