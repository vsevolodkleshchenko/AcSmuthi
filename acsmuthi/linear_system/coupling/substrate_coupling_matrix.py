try:
    from acsmuthi.utility.cython_opt import cython_speedups as cysp

    def substrate_coupling_block(receiver_pos, emitter_pos, k, order):
        return cysp.substrate_coupling_block(receiver_pos, emitter_pos, k, order)


except Exception as e:
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


import numpy as np
import scipy.special as ss
import scipy.integrate as si
import acsmuthi.utility.wavefunctions as wvfs
from acsmuthi.utility.mathematics import dec_to_cyl, legendres_table, legendre_prefactor
from acsmuthi.linear_system.coupling.coupling_basics import k_contour, fresnel_r


def substrate_coupling_block_integrate(receiver_pos, emitter_pos, k, order, k_parallel=None, legendres=None):
    block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)

    dist = receiver_pos - emitter_pos
    d_rho, d_phi, d_z = dec_to_cyl(dist[0], dist[1], dist[2])
    ds = np.abs(emitter_pos[2])

    if k_parallel is None:
        k_p = k_contour(
            k_start_deflection=0.9,
            k_stop_deflection=1.1,
            dk_imag_deflection=1e-2,
            k_finish=3,
            dk=1e-2
        ) * k
    else:
        k_p = k_parallel
    k_z = np.emath.sqrt(k ** 2 - k_p ** 2)

    if legendres is None:
        legendres = legendres_table(k_z / k, order)

    for m, n in wvfs.multipoles(order):
        i_mn = n ** 2 + n + m
        leg_norm_mn = (legendres[0][m, n] if m >= 0 else legendres[1][-m, n]) * legendre_prefactor(m, n)

        for mu, nu in wvfs.multipoles(order):
            i_munu = nu ** 2 + nu + mu
            leg_norm_munu = (legendres[0][mu, nu] if mu >= 0 else legendres[1][-mu, nu]) * legendre_prefactor(mu, nu)
            leg_norm_munu = leg_norm_munu if (nu + mu) % 2 == 0 else - leg_norm_munu

            integrand = fresnel_r(k_p) * np.exp(1j * k_z * (2 * ds + d_z)) * k_p / k_z * \
                ss.jn(mu - m, k_p * d_rho) * leg_norm_mn * leg_norm_munu
            integral = si.trapz(integrand, k_p / k)

            block[i_mn, i_munu] = 4 * np.pi * 1j ** (n - nu + mu - m) * np.exp(1j * (mu - m) * d_phi) * integral
    return block
