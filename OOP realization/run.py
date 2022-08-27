import numpy as np
import classes as cls
import time
import rendering
import cross_sections as cs
import oop_cross_sections as oop_cs
import fields
import oop_fields
import forces
from layers import Layer
from particles import Particle
from linear_system import LinearSystem


def build_new_ps_2s():
    r"""Builds physical system with 2 spheres"""
    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 6

    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    spheres = np.array([sphere1, sphere2])

    ps = cls.System(fluid, incident_field, spheres)
    return ps


def build_new_ps_2s_l():
    r"""Builds physical system with 2 spheres and 1 interface(layer)"""
    # parameters of incident field
    direction = np.array([-1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 6

    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    spheres = np.array([sphere1, sphere2])

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, 2
    ro_interface = ro_sph
    c_interface = c_sph
    interface = Layer(ro_interface, c_interface, a, b, c, d)

    ps = cls.System(fluid, incident_field, spheres, interface)
    return ps


def new_compute(physical_system, order, cross_sections_on=False, forces_on=False, slice_field_on=False,
            bound=None, number_of_points=None, plane=None, plane_number=None):
    """Function that present the results and timing"""
    linear_system = LinearSystem(physical_system, order)
    t_start = time.process_time()
    t_cs_start = t_cs_finish = t_f_start = t_f_finish = t_sf_start = t_sf_finish = 0
    linear_system.solve()
    solution_coefficients = linear_system.solution_coefficients
    t_solution = time.process_time()
    if cross_sections_on:
        t_cs_start = time.process_time()
        print("Scattering and extinction cs:", *oop_cs.cross_section(physical_system, order))
        t_cs_finish = time.process_time()
    if forces_on:
        t_f_start = time.process_time()
        print("Forces:\n", forces.all_forces(solution_coefficients, physical_system, order))
        t_f_finish = time.process_time()
    if slice_field_on:
        t_sf_start = time.process_time()
        span = rendering.build_discretized_span(bound, number_of_points)
        x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
        tot_field = np.real(oop_fields.compute_total_field(physical_system, x_p, y_p, z_p))
        # tot_field = np.real(
        #     fields.total_field(x_p, y_p, z_p, solution_coefficients, physical_system, order, incident_field_on=True))
        rendering.slice_plot(tot_field, x_p, y_p, z_p, span_v, span_h, physical_system, plane=plane)
        t_sf_finish = time.process_time()
    print("Solving the system:", t_solution - t_start, "s")
    print("Counting cross sections:", t_cs_finish - t_cs_start, "s")
    print("Counting forces:", t_f_finish - t_f_start, "s")
    print("Counting fields and drawing the slice plot:", t_sf_finish - t_sf_start, "s")
    print("All process:", time.process_time() - t_start, "s")


def simulation():
    r"""Main simulation function that start computations"""
    bound, number_of_points = 6, 201

    physical_system = build_new_ps_2s()

    order = 6

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    # compute(physical_system, order, cross_sections_on=True, forces_on=True)
    new_compute(physical_system, order, cross_sections_on=True, forces_on=True, slice_field_on=True, bound=bound,
            number_of_points=number_of_points, plane=plane, plane_number=plane_number)


simulation()
