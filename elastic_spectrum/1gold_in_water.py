from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np
import csv


def write_csv(data, fieldnames, filename):
    with open(".\\spectrums_csv\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def one_golden_spheres_in_water(ka):
    order = 6
    p0, rho_fluid, c_fluid = 1, 997, 1403
    direction = np.array([0.5, 0.5, -0.70711])
    r_sph, rho_sph, c_sph_l, c_sph_t = 1, 19300, 3240, 1200
    freq = (ka / r_sph * c_fluid) / (2 * np.pi)
    incident_field = PlaneWave(k=ka / r_sph, amplitude=p0, direction=direction)
    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid)
    sphere = SphericalParticle(np.array([0., 0, 0]), r_sph, rho_sph, c_sph_l, order, c_sph_t)
    particles = np.array([sphere])
    sim = Simulation(particles, fluid, incident_field, freq, order)
    return sim


def count_spectrum(sim_func, name):
    ka_frequencies = np.linspace(0.5, 4.5, 150, dtype=float)
    spectrum_table = np.zeros((len(ka_frequencies), 5))
    spectrum_table[:, 0] = ka_frequencies
    for i, ka in enumerate(ka_frequencies):
        print("Frequency:", i, "of", len(ka_frequencies))
        sim = sim_func(ka)
        sim.run()
        spectrum_table[i, 1] = cs.extinction_cs(sim)
        spectrum_table[i, 2:] = forces.all_forces(sim)[0]
    header = ["ka", "cs", "fx", "fy", "fz"]
    write_csv(spectrum_table, header, name)


# count_spectrum(one_golden_spheres_in_water, 'one_golden_water_sap')
# count_spectrum(one_golden_spheres_in_water, 'one_golden_water_sil')
