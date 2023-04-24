import numpy as np

from acsmuthi.utility import separation_coefficients as seps, wavefunctions as wvfs


def test_j_additional_theorem():
    freq = 82  # [Hz]
    c = 331  # [m/s]
    k = 2 * np.pi * freq / c
    pos1, pos2 = np.array([0, 0, -1]), np.array([0, 0, 1])

    # coordinates
    x_p, z_p = np.meshgrid(np.linspace(-7, 7, 200), np.linspace(-7, 7, 200))
    y_p = np.full_like(x_p, 0.)

    # order of decomposition
    order = 13

    # main part
    m, n = 1, 1
    dist = pos1 - pos2
    desired_j = wvfs.regular_wvf(m, n, x_p + dist[0], y_p + dist[1], z_p + dist[2], k)

    srw_array = np.zeros(((order+1) ** 2, *x_p.shape), dtype=complex)
    for mu, nu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
        i = nu ** 2 + nu + mu
        srw_array[i] = wvfs.regular_wvf(mu, nu, x_p, y_p, z_p, k) * \
                       seps.regular_separation_coefficient(m, mu, n, nu, k, dist)

    actual_j = np.sum(srw_array, axis=0)
    np.testing.assert_allclose(actual_j, desired_j, rtol=1e-2)
