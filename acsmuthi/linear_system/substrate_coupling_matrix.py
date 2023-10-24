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


import numpy as np
import scipy.special as ss
import scipy.integrate as si
from acsmuthi.utility import wavefunctions as wvfs
from acsmuthi.utility.separation_coefficients import gaunt_coefficient
from acsmuthi.utility.mathematics import dec_to_cyl
from acsmuthi.utility.legendres import legendres_table


def fresnel_r(k_parallel):
    return 1


def k_contour(k_start_deflection, k_stop_deflection, dk_imag_deflection, k_finish, dk):
    if k_start_deflection is None:
        return np.arange(0, k_finish, dk)

    path_pieces = []

    if k_start_deflection != 0:
        start_path = np.arange(0, k_start_deflection, dk)
    else:
        start_path = 0 - 1j * np.arange(0, dk_imag_deflection, dk)
    path_pieces.append(start_path)

    if k_stop_deflection is not None:
        deflected_path = np.arange(k_start_deflection, k_stop_deflection, dk) - 1j * dk_imag_deflection
        deflection_stop_path = k_stop_deflection + 1j * np.arange(-dk_imag_deflection, 0, dk)
        finish_path = np.arange(k_stop_deflection, k_finish, dk)
        path_pieces.extend([deflected_path, deflection_stop_path, finish_path])
    else:
        deflected_path = np.arange(k_start_deflection, k_finish, dk) - 1j * dk_imag_deflection
        path_pieces.append(deflected_path)

    return np.concatenate(path_pieces)


def substrate_coupling_block_integrate(receiver_pos, emitter_pos, k, order, k_parallel=None, legendres=None):
    block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)

    dist = receiver_pos - emitter_pos
    d_rho, d_phi, d_z = dec_to_cyl(dist[0], dist[1], dist[2])
    ds = np.abs(emitter_pos[2])

    if k_parallel is None:
        k_p = k_contour(
            k_start_deflection=k-0.1,
            k_stop_deflection=k+0.1,
            dk_imag_deflection=0.001,
            dk=0.0005,
            k_finish=10
        )
    else:
        k_p = k_parallel
    k_z = np.emath.sqrt(k ** 2 - k_p ** 2)

    if legendres is None:
        legendres = legendres_table(k_z / k, order)

    for m, n in wvfs.multipoles(order):
        i_mn = n ** 2 + n + m

        clp_mn = legendres[0][m, n] if m >= 0 else legendres[1][-m, n]
        prefactor_mn = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
        leg_norm_mn = prefactor_mn * clp_mn

        for mu, nu in wvfs.multipoles(order):
            i_munu = nu ** 2 + nu + mu

            clp_munu = legendres[0][mu, nu] if mu >= 0 else legendres[1][-mu, nu]
            prefactor_munu = np.sqrt((2 * nu + 1) / 4 / np.pi * ss.factorial(nu - mu) / ss.factorial(nu + mu))
            leg_norm_munu = prefactor_munu * clp_munu if (nu + mu) % 2 == 0 else - prefactor_munu * clp_munu

            integrand = fresnel_r(k_p) * np.exp(1j * k_z * (2 * ds + d_z)) * k_p / k_z * \
                ss.jn(mu - m, k_p * d_rho) * leg_norm_mn * leg_norm_munu
            integral = si.trapz(integrand, k_p)

            coefficient = 4 * np.pi * 1j ** (n - nu + mu - m) / k
            block[i_mn, i_munu] = coefficient * np.exp(1j * (mu - m) * d_phi) * integral
    return block


# def check_integrator():
#     import matplotlib.pyplot as plt
#
#     m, n, mu, nu = 3, 3, 3, 3
#     k, pos1, pos2 = 1, np.array([-2, 0, 2]), np.array([-2, 2, 4])
#
#     k_waypoint = np.linspace(0.00005, 0.01, 30)
#
#     els1, els2 = [], []
#     for k_tested in k_waypoint:
#         k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k+0.1, dk_imag_deflection=0.01, k_finish=10, dk=k_tested)
#         coup = substrate_coupling_block_integrate(pos2, pos1, k, max(n, nu), k_parallel)
#         els1.append(coup[nu ** 2 + nu + mu][n ** 2 + n + m])
#         k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k+0.1, dk_imag_deflection=0.001, k_finish=10, dk=k_tested)
#         coup = substrate_coupling_block_integrate(pos2, pos1, k, max(n, nu), k_parallel)
#         els2.append(coup[nu ** 2 + nu + mu][n ** 2 + n + m])
#     # show_contour(k_parallel)
#
#     true_el = substrate_coupling_element(m, n, mu, nu, k, pos1, pos2)
#     err1, err2 = np.abs((np.array(els1) - true_el)), np.abs((np.array(els2) - true_el))
#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.loglog(k_waypoint, err1, linewidth=3, linestyle='-.')
#     ax.loglog(k_waypoint, err2, linewidth=3, linestyle='--')
#     plt.show()
#
#
# # check_integrator()
