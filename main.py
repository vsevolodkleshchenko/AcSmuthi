import numpy as np
import time
import rendering
import postprocessing as pp


def simulation():
    r""" main simulation function """
    # coordinates
    number_of_points = 200
    l = 10
    span_x = np.linspace(-l, l, number_of_points)
    span_y = np.linspace(-l, l, number_of_points)
    span_z = np.linspace(-l, l, number_of_points)
    span = np.array([span_x, span_y, span_z])

    # parameters of fluid
    freq = 82
    ro = 1.225
    c_f = 331
    k_fluid = 2 * np.pi * freq / c_f

    # parameters of the spheres
    c_sph = 1403
    k_sph = 2 * np.pi * freq / c_sph
    r_sph = 1
    ro_sph = 1050
    sphere = np.array([k_sph, r_sph, ro_sph])
    spheres = np.array([sphere, sphere])

    # parameters of configuration
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    poses = np.array([pos1, pos2])

    # parameters of the field
    k_x = 0.70711 * k_fluid
    k_y = 0
    k_z = 0.70711 * k_fluid
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 5

    print("Scattering and extinction cross section:", *pp.cross_section(k, ro, poses, spheres, order))

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    tot_field = np.real(pp.total_field(x_p, y_p, z_p, k, ro, poses, spheres, order))
    rendering.slice_plot(tot_field, x_p, y_p, z_p, span_v, span_h, poses, spheres, plane=plane)


def timetest(simulation):
    start = time.process_time()
    simulation()
    end = time.process_time()
    print("Time:", end-start)


timetest(simulation)
