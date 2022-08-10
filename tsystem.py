import numpy as np
import scipy
import scipy.special as ss
import wavefunctions as wvfs
import mathematics as mths


def scaled_coefficient(n, sph, ps):
    r""" scaled coefficient - eq(21) in lopes2016 """
    k_q = ps.k_spheres[sph]
    rho_0 = ps.fluid.rho
    k = ps.k_fluid
    rho_q = ps.spheres[sph].rho
    a_q = ps.spheres[sph].r
    gamma_q = k_q * rho_0 / k / rho_q
    s1 = np.zeros((2, 2), dtype=complex)
    s2 = np.zeros((2, 2), dtype=complex)

    s1[0, 0] = gamma_q * scipy.special.spherical_jn(n, k * a_q)
    s1[0, 1] = ss.spherical_jn(n, k_q * a_q)
    s1[1, 0] = ss.spherical_jn(n, k * a_q, derivative=True)
    s1[1, 1] = ss.spherical_jn(n, k_q * a_q, derivative=True)

    s2[0, 0] = - gamma_q * mths.sph_hankel1(n, k * a_q)
    s2[0, 1] = ss.spherical_jn(n, k_q * a_q)
    s2[1, 0] = - mths.sph_hankel1_der(n, k * a_q)
    s2[1, 1] = ss.spherical_jn(n, k_q * a_q, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def system_matrix(ps, order):
    r""" build matrix of system like in eq(20) in lopes2016 """
    t_matrix = np.zeros((ps.num_sph, ps.num_sph, (order+1)**2, (order+1)**2), dtype=complex)
    all_spheres = np.arange(ps.num_sph)
    for sph in all_spheres:
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            t_matrix[sph, sph, imn, imn] = 1 / scaled_coefficient(n, sph, ps)
            other_spheres = np.where(all_spheres != sph)[0]
            for osph in other_spheres:
                for mu, nu in wvfs.multipoles(order):
                    imunu = nu ** 2 + nu + mu
                    distance = - ps.spheres[osph].pos + ps.spheres[sph].pos
                    t_matrix[sph, osph, imn, imunu] = - wvfs.outgoing_separation_coefficient(mu, m, nu, n, ps.k_fluid,
                                                                                             distance)
    t_matrix2d = np.concatenate(np.concatenate(t_matrix, axis=1), axis=1)
    print(np.linalg.det(t_matrix2d))
    return t_matrix2d


def system_rhs(ps, order):
    r""" build right hand side of system like in eq(20) in lopes2016 """
    rhs = np.zeros((ps.num_sph, (order+1)**2), dtype=complex)
    for sph in range(ps.num_sph):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            rhs[sph, imn] = wvfs.local_incident_coefficient(m, n, ps.k_fluid, ps.incident_field.dir,
                                                            ps.spheres[sph].pos, order)
    rhs1d = np.concatenate(rhs)
    return rhs1d


def solve_system(ps, order):
    r""" solve system like in eq(20) in lopes2016 """
    sc_coef1d = scipy.linalg.solve(system_matrix(ps, order), system_rhs(ps, order))
    sc_coef = sc_coef1d.reshape((ps.num_sph, (order + 1) ** 2))
    in_coef = np.zeros((ps.num_sph, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            in_coef[sph, imn] = (ss.spherical_jn(n, ps.k_fluid * ps.spheres[sph].r) / scaled_coefficient(n, sph, ps) +
                                 mths.sph_hankel1(n, ps.k_fluid * ps.spheres[sph].r)) * sc_coef[sph, imn] / \
                                ss.spherical_jn(n, ps.k_spheres[sph] * ps.spheres[sph].r)
    return sc_coef, in_coef


def effective_incident_coefficients(sph, sc_coef, ps, order):
    r""" build np.array of effective incident coefficients for all n <= order """
    ef_inc_coef = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        ef_inc_coef[imn] = sc_coef[imn] / scaled_coefficient(n, sph, ps)
    return ef_inc_coef


def d_matrix(ps, order):
    """ build D matrix from report - matrix of S coefficients """
    d = np.zeros((ps.num_sph, (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(order):
                imunu = nu ** 2 + nu + mu
                d[sph, imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, ps.k_fluid, ps.spheres[sph].pos)
    d_2d = d.reshape((ps.num_sph * (order + 1) ** 2, (order + 1) ** 2))
    return d_2d


def r_matrix(ps, order):
    d = np.zeros((ps.num_sph, (order + 1) ** 2, (order + 1) ** 2), dtype=complex)

