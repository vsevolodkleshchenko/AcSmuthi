import numpy as np
from postprocessing import rendering
from utility import mathematics as mths, wavefunctions as wvfs


freq = 82  # [Hz]
c = 331  # [m/s]
k = 2 * np.pi * freq / c
pos1, pos2 = np.array([0, 0, -1]), np.array([0, 0, 1])

# coordinates
bound, number_of_points = 7, 200
span = rendering.build_discretized_span(bound, number_of_points)
plane = 'xz'
plane_number = int(number_of_points / 2) + 1
x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)

# order of decomposition
order = 13


def j_additional_theorem_test(m, n):
    dist = pos1 - pos2
    desired_j = wvfs.regular_wvf(m, n, x_p + dist[0], y_p + dist[1], z_p + dist[2], k)

    srw_array = np.zeros(((order+1) ** 2, len(x_p)), dtype=complex)
    for mu, nu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
        i = nu ** 2 + nu + mu
        srw_array[i] = wvfs.regular_wvf(mu, nu, x_p, y_p, z_p, k) * \
                       wvfs.regular_separation_coefficient(m, mu, n, nu, k, dist)

    actual_j = mths.multipoles_fsum(srw_array, len(x_p))
    rendering.plots_for_tests(actual_j, desired_j, span_v, span_h)
    np.testing.assert_allclose(actual_j, desired_j, rtol=1e-2)


j_additional_theorem_test(1, 1)
