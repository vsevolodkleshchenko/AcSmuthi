import numpy as np
import physical_systems as phs
from postprocessing import rendering
from utility import mathematics as mths, wavefunctions as wvfs

ps = phs.build_ps_2s()

# coordinates
bound, number_of_points = 5, 201
span = rendering.build_discretized_span(bound, number_of_points)

# order of decomposition
order = 20

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)


def h_additional_theorem_test(m, n):
    pos0, pos1 = ps.spheres[0].pos, ps.spheres[1].pos
    dist = pos0 - pos1
    desired_h = wvfs.outgoing_wvf(m, n, x_p - pos1[0], y_p - pos1[1], z_p - pos1[2], ps.k_fluid)
    sow_array = np.zeros(((order+1) ** 2, len(x_p)), dtype=complex)
    i = 0
    for mu, nu in wvfs.multipoles(order):
        sow_array[i] = wvfs.regular_wvf(mu, nu, x_p - pos0[0], y_p - pos0[1], z_p - pos0[2], ps.k_fluid) * \
                       wvfs.outgoing_separation_coefficient(m, mu, n, nu, ps.k_fluid, dist)
        i += 1
    actual_h = mths.multipoles_fsum(sow_array, len(x_p))
    actual_h = rendering.draw_spheres(actual_h, ps, x_p, y_p, z_p)
    desired_h = rendering.draw_spheres(desired_h, ps, x_p, y_p, z_p)
    rx, ry, rz = x_p - pos0[0], y_p - pos0[1], z_p - pos0[2]
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    actual_h = np.where(r >= 0.6 * np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2), 0, actual_h)
    desired_h = np.where(r >= 0.6 * np.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2), 0, desired_h)
    rendering.plots_for_tests(actual_h, desired_h, span_v, span_h)
    rendering.slice_plot(np.abs(actual_h - desired_h), x_p, y_p, z_p, span_v, span_h, ps)
    np.testing.assert_allclose(actual_h, desired_h, rtol=1e-2)


# def h_test():
#     i = 0
#     for munu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
#         print(munu)
#         print(wvfs.outgoing_wave_function(munu[0], munu[1], 0.000001, 0.000001, 0.000001, k_l))
#         h = wvfs.outgoing_wave_function(munu[0], munu[1], x_p, y_p, z_p, k_l)
#         rendering.plots_for_tests(h, h)
#         i += 1


h_additional_theorem_test(1, 1)
