import numpy as np
import time
import rendering
import postprocessing as pp
import classes as cls


def time_test(sim):
    start = time.process_time()
    sim()
    end = time.process_time()
    print("Time:", end-start)


def simulation():
    r""" main simulation function """
    # coordinates
    bound, number_of_points = 10, 201
    span = rendering.build_discretized_span(bound, number_of_points)

    physical_system = cls.build_ps()

    order = 10
    print(pp.forces(physical_system, order))
    print("Scattering and extinction cross section:", *pp.cross_section(physical_system, order))

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    print(y_p)
    tot_field = np.real(pp.total_field(x_p, y_p, z_p, physical_system, order))
    rendering.slice_plot(tot_field, x_p, y_p, z_p, span_v, span_h, physical_system, plane=plane)


time_test(simulation)
