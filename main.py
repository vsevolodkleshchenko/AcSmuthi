import numpy as np
import time
import rendering
import tsystem
import cross_sections as cs
import fields
import forces
import physical_systems as phs


def compute(physical_system, order, cross_sections_on=False, forces_on=False, slice_field_on=False,
            bound=None, number_of_points=None, plane=None, plane_number=None):
    """Function that present the results and timing"""
    t_start = time.process_time()
    t_cs_start = t_cs_finish = t_f_start = t_f_finish = t_sf_start = t_sf_finish = 0
    if physical_system.interface:
        solution_coefficients = tsystem.solve_layer_system(physical_system, order)
    else:
        solution_coefficients = tsystem.solve_system(physical_system, order)
    t_solution = time.process_time()
    if cross_sections_on:
        t_cs_start = time.process_time()
        print("Scattering and extinction cs:", *cs.cross_section(solution_coefficients, physical_system, order))
        t_cs_finish = time.process_time()
    if forces_on:
        t_f_start = time.process_time()
        print("Forces:\n", forces.all_forces(solution_coefficients, physical_system, order))
        t_f_finish = time.process_time()
    if slice_field_on:
        t_sf_start = time.process_time()
        span = rendering.build_discretized_span(bound, number_of_points)
        x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
        tot_field = np.real(
            fields.total_field(x_p, y_p, z_p, solution_coefficients, physical_system, order, incident_field_on=True))
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

    physical_system = phs.build_ps_2s()

    order = 6

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    # compute(physical_system, order, cross_sections_on=True, forces_on=True)
    compute(physical_system, order, cross_sections_on=True, forces_on=True, slice_field_on=True, bound=bound,
            number_of_points=number_of_points, plane=plane, plane_number=plane_number)


simulation()
