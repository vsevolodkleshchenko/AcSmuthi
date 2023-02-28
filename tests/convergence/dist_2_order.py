import numpy as np
import csv
from multiprocessing import Process, Queue

from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.postprocessing import cross_sections as cs
from acsmuthi.postprocessing import forces

from acsmuthi.initial_field import PlaneWave
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium


def silica_aerogel_sphere_in_standing_wave_ls(order, dist):
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    direction = np.array([0, 0, 1])
    freq = 1400  # [Hz]
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    ro_sph = 80  # [kg/m^3]
    r_sph = 0.01  # 0.001
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(ro_fluid, c_fluid)

    pos1, pos2 = np.array([-dist/2, 0, 0]), np.array([dist/2, 0, 0])
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1, sphere2])

    ls = LinearSystem(particles, fluid, incident_field, freq, order, True)
    return ls


def sub_proc(s, param, proc_result):
    lin_sys = silica_aerogel_sphere_in_standing_wave_ls(s, param)
    lin_sys.prepare()
    lin_sys.solve()
    ecs = cs.extinction_cs(lin_sys.particles, lin_sys.medium, lin_sys.incident_field, 12000)
    all_forces = forces.all_forces(lin_sys.particles, lin_sys.medium, lin_sys.incident_field)
    proc_result.put((ecs, np.concatenate(all_forces)))


def main_proc(sample, param, tab_len):
    table = np.zeros((len(sample), tab_len), dtype=float)
    for i, s in enumerate(sample):
        print("     Order:", i, "of", len(sample) - 1)
        queue_res = Queue()
        sub_process = Process(target=sub_proc, args=(s, param, queue_res,))
        sub_process.start()
        ecs, frcs = queue_res.get()
        table[i, 0] = ecs
        table[i, 1:] = frcs
        sub_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\distance2_csv\\Dl0.1\\" + filename + ".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    orders = np.arange(6, 19)
    distances = np.linspace(0.022, 0.4, 21)  # l = 0.0236, 0.236, f = 14000, 1400, 0.1, 0.4
    header = ["ord", "ecs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z"]
    tot_table = np.zeros((len(orders), len(header)))
    tot_table[:, 0] = orders[:]

    for i, d in enumerate(distances):
        print(i, "distance of", len(distances)-1)
        tot_table[:, 1:] = main_proc(orders, d, len(header) - 1)
        write_csv(tot_table, header, "order2sph"+str(d / 0.236))
