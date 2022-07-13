import numpy as np
import time
import rendering
import postprocessing as pp


def simulation():
    r""" main simulation function """
    # coordinates
    bound, number_of_points = 10, 200
    span = rendering.build_discretized_span(bound, number_of_points)

    # parameters of fluid
    freq = 82
    ro_fluid = 1.225
    c_fluid = 331
    k_fluid = 2 * np.pi * freq / c_fluid

    # parameters of the spheres
    c_sph = 1403
    k_sph = 2 * np.pi * freq / c_sph
    r_sph = 1
    ro_sph = 1050
    sphere = np.array([k_sph, r_sph, ro_sph])
    spheres = np.array([sphere])

    # parameters of configuration
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([0, 0, 2.5])
    positions = np.array([pos1])

    # parameters of the field
    k_x = 0  # 0.70711 * k_fluid
    k_y = 0
    k_z = k_fluid  # 0.70711 * k_fluid
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 10

    print("Scattering and extinction cross section:", *pp.cross_section(k, ro_fluid, positions, spheres, order))

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    tot_field = np.real(pp.total_field(x_p, y_p, z_p, k, ro_fluid, positions, spheres, order))
    rendering.slice_plot(tot_field, x_p, y_p, z_p, span_v, span_h, positions, spheres, plane=plane)


def time_test(sim):
    start = time.process_time()
    sim()
    end = time.process_time()
    print("Time:", end-start)


time_test(simulation)
