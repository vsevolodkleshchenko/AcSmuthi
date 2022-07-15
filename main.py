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
    spheres = np.array([sphere])

    # parameters of configuration
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([0, 0, 2.5])
    positions = np.array([pos1])

    # parameters of the field
    p0 = 1  # [kg/m/s^2] = [Pa]
    I_inc = p0 ** 2 / (2 * ro_fluid * c_fluid)  # [W/m^2]
    k_x = 0.70711 * k_fluid
    k_y = 0
    k_z = 0.70711 * k_fluid
    k = np.array([k_x, k_y, k_z])

    # order of decomposition
    order = 10
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


time_test(simulation)


########################################################################################################################


def build_ps():
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
    pos1 = np.array([0, 0, 0])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    sphere_cls = cls.Sphere(pos1, r_sph, ro_sph, c_sph)
    spheres_cls = np.array([sphere_cls])

    ps = cls.System(fluid, incident_field, spheres_cls)
    return ps


def simulation_cls():
    r""" main simulation function """
    # coordinates
    bound, number_of_points = 10, 200
    span = rendering.build_discretized_span(bound, number_of_points)

    ps = build_ps()

    order = 10
    print("Scattering and extinction cross section:", *pp.cross_section_cls(ps, order))

    # plane = 'xz'
    # plane_number = int(number_of_points / 2) + 1
    #
    # x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
    # tot_field = np.real(pp.total_field(x_p, y_p, z_p, k, ro_fluid, positions, spheres, order))
    # rendering.slice_plot(tot_field, x_p, y_p, z_p, span_v, span_h, positions, spheres, plane=plane)
