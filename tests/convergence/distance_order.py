import numpy as np
import csv

from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def silica_aerogel_sphere_in_standing_wave_ls(order, dist):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    freq = 18.8  #
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    ro_sph = 80  # [kg/m^3]
    r_sph = 1
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(ro_fluid, c_fluid)

    pos1, pos2 = np.array([-dist/2, 0, 0]), np.array([dist/2, 0, 0])
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1, sphere2])

    sim = Simulation(particles, fluid, incident_field, freq, order, True)
    sim.run()
    ecs = cs.extinction_cs(particles, fluid, incident_field, freq)
    all_forces = forces.all_forces(particles, fluid, incident_field)
    return ecs, np.concatenate(all_forces)


def main_proc(orders, distance):
    table = np.zeros((len(orders), 7), dtype=float)
    for i, order in enumerate(orders):
        print("     Order:", i, "of", len(orders))
        ecs, frcs = silica_aerogel_sphere_in_standing_wave_ls(order, distance)
        table[i, 0] = ecs
        table[i, 1:] = frcs
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\distance_order_csv\\Dl0_11\\" + filename + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def main():
    orders = np.arange(1, 14)
    lda = 331 / 18.8
    distances = np.linspace(2.2, 4 * lda, 31)
    header = ["ord", "ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]

    for i, d in enumerate(distances):
        print(i, "distance of", len(distances)-1)
        tot_table[:, 1:] = main_proc(orders, d)
        write_csv(tot_table, header, "2sph_dl_"+str(np.round(d / lda, 2)))


main()
