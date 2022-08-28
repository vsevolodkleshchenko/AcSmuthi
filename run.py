import numpy as np
from oop_fields import PlaneWave
from layers import Layer
from particles import Particle
from medium import Fluid
from simulation import Simulation


def build_simulation():
    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    # direction = np.array([0.70711, 0, 0.70711])
    direction = np.array([-1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, 2
    ro_interface = ro_sph
    c_interface = c_sph

    order = 6

    layer = Layer(ro_interface, c_interface, a, b, c, d)
    fluid = Fluid(ro_fluid, c_fluid)
    incident_field = PlaneWave(p0, k, 'regular', order, direction)
    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    particles = np.array([sphere1, sphere2])

    bound, number_of_points = 6, 201
    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    sim = Simulation(particles, fluid, incident_field, freq, order, layer=layer,
                     plane=plane, bound=bound, plane_number=plane_number, number_of_points=number_of_points)
    return sim


def oop_run():
    sim = build_simulation()
    sim.run(cross_sections_flag=True, forces_flag=True, plot_flag=True)


oop_run()
