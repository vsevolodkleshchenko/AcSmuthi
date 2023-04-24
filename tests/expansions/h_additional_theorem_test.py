import numpy as np

import acsmuthi.utility.wavefunctions as wvfs
import acsmuthi.utility.separation_coefficients as seps


def test_h_additional_theorem():
    c = 331  # [m/s]
    freq = 82  # [Hz]
    k = 2 * np.pi * freq / c  # [1/m]
    pos1, pos2 = np.array([0, 0, -2.5]), np.array([0, 0, 2.5])

    # coordinates
    x_p, z_p = np.meshgrid(np.linspace(-5, 5, 201), np.linspace(-5, 5, 201))
    y_p = np.full_like(x_p, 0.)

    # order of decomposition
    order = 20

    # main part
    m, n = 1, 1
    dist = pos1 - pos2
    desired_h = wvfs.outgoing_wvf(m, n, x_p - pos2[0], y_p - pos2[1], z_p - pos2[2], k)

    sow_array = np.zeros(((order+1) ** 2, *x_p.shape), dtype=complex)
    for mu, nu in wvfs.multipoles(order):
        i = nu ** 2 + nu + mu
        sow_array[i] = wvfs.regular_wvf(mu, nu, x_p - pos1[0], y_p - pos1[1], z_p - pos1[2], k) * \
                       seps.outgoing_separation_coefficient(m, mu, n, nu, k, dist)
    actual_h = np.sum(sow_array, axis=0)

    rx1, ry1, rz1, rx2, ry2, rz2 = x_p-pos1[0], y_p-pos1[1], z_p-pos1[2], x_p-pos2[0], y_p-pos2[1], z_p-pos2[2]
    r1, r2 = np.sqrt(rx1 ** 2 + ry1 ** 2 + rz1 ** 2), np.sqrt(rx2 ** 2 + ry2 ** 2 + rz2 ** 2)
    actual_h = np.where((r1 <= 0.5) | (r2 <= 0.5), 0, actual_h)
    desired_h = np.where((r1 <= 0.5) | (r2 <= 0.5), 0, desired_h)

    rx, ry, rz1 = x_p - pos1[0], y_p - pos1[1], z_p - pos1[2]
    r = np.sqrt(rx ** 2 + ry ** 2 + rz1 ** 2)
    actual_h = np.where(r >= 0.6 * np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2), 0, actual_h)
    desired_h = np.where(r >= 0.6 * np.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2), 0, desired_h)

    np.testing.assert_allclose(actual_h, desired_h, rtol=1e-2)
