import numpy as np
import time
import rendering
import postprocessing as pp
import classes as cls


def simulation():
    r""" main simulation function """
    # coordinates
    bound, number_of_points = 10, 200
    span = rendering.build_discretized_span(bound, number_of_points)

    # parameters of fluid
    freq = 82  # [Hz]
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    k_fluid = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    c_sph = 1403  # [m/s]
    k_sph = 2 * np.pi * freq / c_sph  # [1/m]
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    sphere = np.array([k_sph, r_sph, ro_sph])
    spheres = np.array([sphere, sphere])

    # parameters of configuration
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    positions = np.array([pos1, pos2])

    # parameters of the field
    p0 = 1  # [kg/m/s^2] = [Pa]
    I_inc = p0 ** 2 / (2 * ro_fluid * c_fluid)  # [W/m^2]
    k_x = 0.70711 * k_fluid
    k_y = 0
    k_z = 0.70711 * k_fluid
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 6
    print("Scattering and extinction cross section:", *pp.cross_section(k, ro_fluid, positions, spheres, order, p0,
                                                                        I_inc))

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


########################################################################################################################


def simulation_cls():
    r""" main simulation function """
    # coordinates
    bound, number_of_points = 10, 200
    span = rendering.build_discretized_span(bound, number_of_points)

    ps = cls.build_ps()

    order = 6
    print("Scattering and extinction cross section:", *pp.cross_section_cls(ps, order))

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    tot_field = np.real(pp.total_field_cls(x_p, y_p, z_p, ps, order))
    rendering.slice_plot_cls(tot_field, x_p, y_p, z_p, span_v, span_h, ps, plane=plane)


time_test(simulation_cls)
