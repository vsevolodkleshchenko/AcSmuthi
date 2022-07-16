import numpy as np
import wavefunctions as wvfs
import rendering
import mathematics as mths

# coordinates
number_of_points = 200
l = 5
span_x = np.linspace(-l, l, number_of_points)
span_y = np.linspace(-l, l, number_of_points)
span_z = np.linspace(-l, l, number_of_points)
span = np.array([span_x, span_y, span_z])

# parameters of fluid
freq = 82
ro = 1.225
c_f = 331
k_fluid = 2 * np.pi * freq / c_f

# parameters of the field
k_x = 0  # 0.70711 * k_fluid
k_y = 0
k_z = k_fluid  # 0.70711 * k_fluid
k = np.array([k_x, k_y, k_z])

# order of decomposition
order = 5

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)


def h_additional_theorem_test(m, n):
    dist = np.array([0, 0, -5])
    desired_h = wvfs.outgoing_wave_function(m, n, x_p + dist[0], y_p + dist[1], z_p + dist[2], k)
    sow_array = np.zeros(((order+1) ** 2, len(x_p)), dtype=complex)
    i = 0
    for munu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
        sow_array[i] = wvfs.outgoing_wave_function(munu[0], munu[1], x_p, y_p, z_p, k) * \
                       wvfs.regular_separation_coefficient(m, munu[0], n, munu[1], k, dist)
        i += 1
    actual_h = mths.multipoles_fsum(sow_array, len(x_p))
    # actual_h = sphrs.draw_spheres(actual_h, np.array([np.array([0, 0, 0])]), spheres, x_p, y_p, z_p)
    rendering.plots_for_tests(actual_h, desired_h, span_v, span_h)
    np.testing.assert_allclose(actual_h, desired_h, rtol=1e-2)


def h_test():
    i = 0
    for munu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
        print(munu)
        print(wvfs.outgoing_wave_function(munu[0], munu[1], 0.000001, 0.000001, 0.000001, k))
        h = wvfs.outgoing_wave_function(munu[0], munu[1], x_p, y_p, z_p, k)
        rendering.plots_for_tests(h, h)
        i += 1


h_additional_theorem_test(1, 1)
