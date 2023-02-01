import numpy as np
from acsmuthi.fields_expansions import PlaneWave, StandingWave
from acsmuthi.layers import Layer
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.simulation import Simulation


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
    ro_sph = 997  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 10

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1, sphere2])

    bound, number_of_points = 4.5, 301
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number)
    return sim


def five_fluid_spheres():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([-1, 0, 0])
    freq = 150  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([1.56, 0, -3.6])
    pos2 = np.array([1.56, 0, 0])
    pos3 = np.array([1.56, 0, 3.6])
    pos4 = np.array([-1.56, 0, -1.8])
    pos5 = np.array([-1.56, 0, 1.8])
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 9

    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    sphere3 = Particle(pos3, r_sph, ro_sph, c_sph, order)
    sphere4 = Particle(pos4, r_sph, ro_sph, c_sph, order)
    sphere5 = Particle(pos5, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1, sphere2, sphere3, sphere4, sphere5])

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


def silica_aerogel_sphere_in_standing_wave():
    # https://doi.org/10.1016/j.jnoncrysol.2018.07.021

    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 334  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 5000  # [Hz]
    p0 = 10000  # [kg/m/s^2] = [Pa]
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

    order = 6

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1])

    bound, number_of_points = 0.04, 151
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number)
    return sim


def two_silica_aerogel_sphere_in_standing_wave():
    # https://doi.org/10.1016/j.jnoncrysol.2018.07.021

    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 334  # [m/s]

    # parameters of incident field
    direction = np.array([0, 0, 1])
    freq = 8275  # [Hz]
    p0 = 10000  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    poisson = 0.12
    young = 197920
    g = 0.5 * young / (1 + poisson)
    pos1 = np.array([0, 0, 0.03])  # [m]
    pos2 = np.array([0, 0, -0.03])
    r_sph = 0.01  # [m]
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    order = 8

    incident_field = StandingWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
    particles = np.array([sphere1, sphere2])

    bound, number_of_points = 0.04, 151
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number)
    return sim


def one_fluid_sphere_above_interface():
    # parameters of medium
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, -0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([1.6, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph = 1403  # [m/s]

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, 0
    ro_interface = ro_sph
    c_interface = c_sph

    order = 12

    layer = Layer(ro_interface, c_interface, a, b, c, d)
    incident_field = PlaneWave(p0, k_l, np.array([0, 0, 0]), 'regular', order, direction)
    fluid = Medium(ro_fluid, c_fluid, incident_field=incident_field)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1])

    bound, number_of_points = 4, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, freq, order, bound=bound, number_of_points=number_of_points, plane=plane,
                     plane_number=plane_number, layer=layer)
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

    order = 11

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
