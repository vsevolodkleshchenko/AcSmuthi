import numpy as np
from fields_expansions import PlaneWave
from layers import Layer
from particles import Particle
from medium import Medium
from simulation import Simulation


def one_fluid_sphere():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([0, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 8

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1])

    bound, number_of_points = 6, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number)
    return sim


def two_fluid_spheres():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 10

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1, sphere2])

    bound, number_of_points = 6, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number)
    return sim


def one_elastic_sphere():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([0, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 2700  # [kg/m^3]
    c_sph_l = 6197  # [m/s]
    c_sph_t = 3122  # [m/s]

    order = 8

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1])

    bound, number_of_points = 6, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number)
    return sim


def two_fluid_spheres_above_interface():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([-1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([0, 0, -2])
    pos2 = np.array([0, 0, 2])
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph = 1403  # [m/s]

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, 1.7
    ro_interface = ro_sph
    c_interface = c_sph

    order = 10

    layer = Layer(ro_interface, c_interface, a, b, c, d)
    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1, sphere2])

    bound, number_of_points = 3.3, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number, layer=layer)
    return sim
