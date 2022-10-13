import numpy as np
from acsmuthi.postprocessing import rendering
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs

c = 331  # [m/s]
freq = 82  # [Hz]
k = 2 * np.pi * freq / c  # [1/m]
pos1, pos2 = np.array([0, 0, -2.5]), np.array([0, 0, 2.5])

# coordinates
bound, number_of_points = 5, 201
span = rendering.build_discretized_span(bound, number_of_points)
plane = 'xz'
plane_number = int(number_of_points / 2) + 1
x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)

# order of decomposition
order = 20


def h_additional_theorem_test(m, n):
    dist = pos1 - pos2
    desired_h = wvfs.outgoing_wvf(m, n, x_p - pos2[0], y_p - pos2[1], z_p - pos2[2], k)

    sow_array = np.zeros(((order+1) ** 2, len(x_p)), dtype=complex)
    for mu, nu in wvfs.multipoles(order):
        i = nu ** 2 + nu + mu
        sow_array[i] = wvfs.regular_wvf(mu, nu, x_p - pos1[0], y_p - pos1[1], z_p - pos1[2], k) * \
                       wvfs.outgoing_separation_coefficient(m, mu, n, nu, k, dist)
    actual_h = mths.multipoles_fsum(sow_array, len(x_p))

    rx1, ry1, rz1, rx2, ry2, rz2 = x_p-pos1[0], y_p-pos1[1], z_p-pos1[2], x_p-pos2[0], y_p-pos2[1], z_p-pos2[2]
    r1, r2 = np.sqrt(rx1 ** 2 + ry1 ** 2 + rz1 ** 2), np.sqrt(rx2 ** 2 + ry2 ** 2 + rz2 ** 2)
    actual_h = np.where((r1 <= 0.5) | (r2 <= 0.5), 0, actual_h)
    desired_h = np.where((r1 <= 0.5) | (r2 <= 0.5), 0, desired_h)

    rx, ry, rz1 = x_p - pos1[0], y_p - pos1[1], z_p - pos1[2]
    r = np.sqrt(rx ** 2 + ry ** 2 + rz1 ** 2)
    actual_h = np.where(r >= 0.6 * np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2), 0, actual_h)
    desired_h = np.where(r >= 0.6 * np.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2), 0, desired_h)

    rendering.plots_for_tests(actual_h, desired_h, span_v, span_h)
    rendering.slice_plot(np.abs(actual_h - desired_h), span_v, span_h)
    np.testing.assert_allclose(actual_h, desired_h, rtol=1e-2)


h_additional_theorem_test(1, 1)
