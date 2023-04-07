import numpy as np
import csv

from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def silica_aerogel_sphere_in_standing_wave_ls(order, r_sph):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0.70711, 0, 0.70711])
    freq = 140  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(ro_fluid, c_fluid)

    pos1 = np.array([0., 0., 0.])
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1])

    lin_sys = LinearSystem(particles, fluid, incident_field, freq, order, True)
    lin_sys.prepare()
    lin_sys.solve()
    ecs = cs.extinction_cs(particles, fluid, incident_field, freq)
    all_forces = forces.all_forces(particles, fluid, incident_field)
    return ecs, np.concatenate(all_forces)


def main_proc(orders, size):
    table = np.zeros((len(orders), 4), dtype=float)
    for i, order in enumerate(orders):
        print("     Order:", i, "of", len(orders))
        ecs, frcs = silica_aerogel_sphere_in_standing_wave_ls(order, size)
        table[i, 0] = ecs
        table[i, 1:] = frcs
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\particle_size_order_csv\\" + filename + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def main():
    orders = np.arange(5, 17)
    lda = 331 / 140
    sizes = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5]) * lda
    header = ["ord", "ecs", "f1x", "f1y", "f1z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]

    for i, r in enumerate(sizes):
        print(i, "size of", len(sizes)-1)
        tot_table[:, 1:] = main_proc(orders, r)
        write_csv(tot_table, header, "Dl_"+str(np.round(2 * r / lda, 2)))


main()
