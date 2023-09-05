import matplotlib, matplotlib.pyplot as plt

from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, rendering, cross_sections as cs
import numpy as np
import csv
import seaborn as sns


plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def write_csv(data, fieldnames, filename):
    with open(".\\spectrums_csv\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def two_golden_spheres_in_water(ka):
    order = 8
    p0, rho_fluid, c_fluid = 1, 1.225, 331
    direction = np.array([0.70711, 0., -0.70711])
    r_sph, rho_sph, c_sph = 1, 997, 1403
    freq = (ka / r_sph * c_fluid) / (2 * np.pi)
    incident_field = PlaneWave(k=ka / r_sph, amplitude=p0, direction=direction)
    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid, is_substrate=True)
    sphere1 = SphericalParticle(position=np.array([-1.5, 0, 1.7]), radius=r_sph, density=rho_sph, pressure_velocity=c_sph, order=order)
    sphere2 = SphericalParticle(position=np.array([1.7, 0., 1.6]), radius=r_sph, density=rho_sph, pressure_velocity=c_sph, order=order)
    particles = np.array([sphere1, sphere2])
    sim = Simulation(particles, fluid, incident_field, freq, order)
    return sim


def count_spectrum(sim_func, name):
    ka_frequencies = np.linspace(0.25, 3.5, 150, dtype=float)
    spectrum_table = np.zeros((len(ka_frequencies), 8))
    spectrum_table[:, 0] = ka_frequencies
    for i, ka in enumerate(ka_frequencies):
        print("Frequency:", i, "of", len(ka_frequencies))
        sim = sim_func(ka)
        sim.run()
        spectrum_table[i, 1] = cs.extinction_cs(sim)
        spectrum_table[i, 2:] = np.concatenate(forces.all_forces(sim))
    header = ["ka", "cs", "f1x", "f1y", "f1z", "f2x", "f2y", "f2z"]
    write_csv(spectrum_table, header, name)


# count_spectrum(two_golden_spheres_in_water, "two_water_in_air_spec")

def draw_spectrum(name):
    colors = sns.color_palette("dark:salmon_r", 4)
    table = np.genfromtxt("spectrums_csv/"+name+".csv", skip_header=1, delimiter=",", dtype=float)
    table_comsol = np.genfromtxt("spectrums_csv/"+name+"_comsol.csv", skip_header=5, delimiter=",", dtype=float)

    ka, ecs = table[:, 0], table[:, 1]
    fx, fz = table[:, 2], table[:, 4]

    ka_comsol = 2 * np.pi * table_comsol[:, 0] / 331
    scs_comsol, ecs_comsol = table_comsol[:, 7], table_comsol[:, 8]
    fx_comsol, fz_comsol = table_comsol[:, 1], table_comsol[:, 3]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(ka, ecs, color=colors[0])
    ax[0].scatter(ka_comsol, ecs_comsol, marker='x', color=colors[3])

    ax[1].plot(ka, fz, color=colors[0])
    ax[1].scatter(ka_comsol, fz_comsol, marker='x', color=colors[3])
    plt.show()


# draw_spectrum("two_water_in_air_spec")
