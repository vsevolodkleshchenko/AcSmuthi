import numpy as np
import wavefunctions as wvfs
import rendering
import mathematics as mths
import classes as cls

# coordinates
bound, number_of_points = 5, 200
span = rendering.build_discretized_span(bound, number_of_points)

direction = np.array([0.70711, 0, 0.70711])
freq = 82  # [Hz]
p0 = 1  # [kg/m/s^2] = [Pa]
incident_field = cls.PlaneWave(direction, freq, p0)

# parameters of fluid
ro_fluid = 1.225  # [kg/m^3]
c_fluid = 331  # [m/s]
fluid = cls.Fluid(ro_fluid, c_fluid)

k_fluid = 2 * np.pi * freq / c_fluid

# order of decomposition
order = 5

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)


def h_additional_theorem_test(m, n):
    dist = np.array([0, 0, -5])
    desired_h = wvfs.outgoing_wvf(m, n, x_p + dist[0], y_p + dist[1], z_p + dist[2], k_fluid)
    sow_array = np.zeros(((order+1) ** 2, len(x_p)), dtype=complex)
    i = 0
    for munu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
        sow_array[i] = wvfs.outgoing_wvf(munu[0], munu[1], x_p, y_p, z_p, k_fluid) * \
                       wvfs.regular_separation_coefficient(m, munu[0], n, munu[1], k_fluid, dist)
        i += 1
    actual_h = mths.multipoles_fsum(sow_array, len(x_p))
    # actual_h = sphrs.draw_spheres(actual_h, np.array([np.array([0, 0, 0])]), spheres, x_p, y_p, z_p)
    rendering.plots_for_tests(actual_h, desired_h, span_v, span_h)
    np.testing.assert_allclose(actual_h, desired_h, rtol=1e-2)


# def h_test():
#     i = 0
#     for munu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
#         print(munu)
#         print(wvfs.outgoing_wave_function(munu[0], munu[1], 0.000001, 0.000001, 0.000001, k))
#         h = wvfs.outgoing_wave_function(munu[0], munu[1], x_p, y_p, z_p, k)
#         rendering.plots_for_tests(h, h)
#         i += 1


h_additional_theorem_test(1, 1)
