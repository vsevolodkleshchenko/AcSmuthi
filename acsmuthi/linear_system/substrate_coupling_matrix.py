try:
    from acsmuthi.utility.cython_opt import cython_speedups as cysp

    def substrate_coupling_block(receiver_pos, emitter_pos, k, order):
        return cysp.substrate_coupling_block(receiver_pos, emitter_pos, k, order)


except Exception as e:
    print("Failed to import cython speedups", str(e))

    import numpy as np
    from acsmuthi.utility import wavefunctions as wvfs
    from acsmuthi.utility.separation_coefficients import gaunt_coefficient


    def substrate_coupling_element(m, n, mu, nu, k, emitter_pos, receiver_pos):
        dist = receiver_pos - emitter_pos
        ds = np.abs(emitter_pos[2])

        dx, dy, dz = dist[0], dist[1], dist[2] + 2 * ds

        if abs(n - nu) >= abs(m - mu):
            q0 = abs(n - nu)
        if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
            q0 = abs(m - mu)
        if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
            q0 = abs(m - mu) + 1
        q_lim = (n + nu - q0) // 2

        sum_array = np.zeros(q_lim + 1, dtype=complex)

        for i, q in enumerate(range(0, q_lim + 1)):
            sum_array[i] = 1j ** (q0 + 2 * q) * wvfs.outgoing_wvf(m - mu, q0 + 2 * q, dx, dy, dz, k) * \
                           gaunt_coefficient(n, m, nu, -mu, q0 + 2 * q)

        return 4 * np.pi * 1j ** (nu - n) * (-1.) ** (n + m + mu) * np.sum(sum_array)


    def substrate_coupling_block(receiver_pos, emitter_pos, k, order):
        block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for mu, nu in wvfs.multipoles(order):
                imunu = nu ** 2 + nu + mu
                block[imn, imunu] = substrate_coupling_element(mu, nu, m, n, k, emitter_pos, receiver_pos)
        return block
