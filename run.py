from acsmuthi.simulation import Simulation
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave as nPlaneWave
import numpy as np


def silica_aerogel_sphere_in_standing_wave():
    # https://doi.org/10.1016/j.jnoncrysol.2018.07.021

    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0, 0, 1])
    freq = 7323.4  # [Hz]   # fs: 3195.8, 4635.7, 6014.3, 7323.4? ; bs: 3089.9, 7209.2?, 9898.6!, 12508.5?
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    pos1 = np.array([0, 0, 0])  # [m]
    r_sph = 0.01  # [m]
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    order = 5

    # incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    incident_field = nPlaneWave(k_l, p0, direction)

    fluid = Medium(ro_fluid, c_fluid)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1])

    bound, number_of_points = 0.045, 151
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, incident_field, freq, order, bound=bound, number_of_points=number_of_points,
                     plane=plane, plane_number=plane_number)
    return sim


simulation = silica_aerogel_sphere_in_standing_wave()
simulation.run(cross_sections_flag=False, forces_flag=True, plot_flag=True)
