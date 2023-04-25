import numpy as np
import csv

from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium


'''
Test to check the convergence of scattering cross sections and acoustic forces for a different distances 
between two spherical particles, depending on the multipole order
'''

# Parameters: steel spheres in air in plane wave
rho_medium, c_medium = 1.225, 331
rho_sph, cp_sph, cs_sph = 7700, 5740, 3092
p0, direction = 1, np.array([0.70711, 0, 0.70711])

# Main parameter for changing scattering regime is radius of the sphere
freq = 50
k, lda = 2 * np.pi * freq / c_medium, c_medium / freq


def main_proc(orders, ka):
    table = np.zeros((len(orders), 4), dtype=float)
    for i, order in enumerate(orders):
        print("     Order:", i, "of", len(orders))

        initial_field = PlaneWave(k=k, amplitude=p0, direction=direction)
        medium = Medium(density=rho_medium, pressure_velocity=c_medium)
        particles = np.array([SphericalParticle(
            position=np.array([0., 0, 0]),
            radius=ka / k,
            density=rho_sph,
            pressure_velocity=cp_sph,
            shear_velocity=cs_sph,
            order=order
        )])
        sim = Simulation(particles, medium, initial_field, freq, order, True)
        sim.run()
        table[i, 0] = cs.extinction_cs(sim.particles, sim.medium, sim.initial_field, freq)
        table[i, 1:] = np.concatenate(forces.all_forces(sim.particles, sim.medium, sim.initial_field))
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\particle_size_order_csv\\" + filename + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def main():
    orders = np.arange(1, 15)
    ka_sizes = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10])
    header = ["ord", "ecs", "f1x", "f1y", "f1z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]

    for i, ka in enumerate(ka_sizes):
        print(i, "size of", len(ka_sizes) - 1)
        tot_table[:, 1:] = main_proc(orders, ka)
        write_csv(tot_table, header, f"ka{ka}")


# main()
