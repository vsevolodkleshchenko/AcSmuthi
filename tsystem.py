import numpy as np
import scipy
import scipy.special
import wavefunctions as wvfs
import mathematics as mths


def scaled_coefficient(n, sph, ps):
    k_q = ps.k_spheres[sph]
    rho_0 = ps.fluid.rho
    k = ps.k_fluid
    rho_q = ps.spheres[sph].rho
    a_q = ps.spheres[sph].r
    gamma_q = k_q * rho_0 / k / rho_q
    s1 = np.zeros((2, 2), dtype=complex)
    s2 = np.zeros((2, 2), dtype=complex)

    s1[0, 0] = gamma_q * scipy.special.spherical_jn(n, k * a_q)
    s1[0, 1] = scipy.special.spherical_jn(n, k_q * a_q)
    s1[1, 0] = scipy.special.spherical_jn(n, k * a_q, derivative=True)
    s1[1, 1] = scipy.special.spherical_jn(n, k_q * a_q, derivative=True)

    s2[0, 0] = - gamma_q * mths.sph_hankel1(n, k * a_q)
    s2[0, 1] = scipy.special.spherical_jn(n, k_q * a_q)
    s2[1, 0] = - mths.sph_hankel1_der(n, k * a_q)
    s2[1, 1] = scipy.special.spherical_jn(n, k_q * a_q, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def system_matrix(ps, order):
    t_matrix = np.zeros((ps.num_sph, ps.num_sph, (order+1)**2, (order+1)**2), dtype=complex)
    all_spheres = np.arange(ps.num_sph)
    for sph in all_spheres:
        for mn in wvfs.multipoles(order):
            imn = mn[1]**2+mn[1]+mn[0]
            t_matrix[sph, sph, imn, imn] = 1 / scaled_coefficient(mn[1], sph, ps)
            other_spheres = np.where(all_spheres != sph)[0]
            for osph in other_spheres:
                for munu in wvfs.multipoles(order):
                    imunu = munu[1]**2+munu[1]+munu[0]
                    distance = ps.spheres[osph].pos - ps.spheres[sph].pos
                    t_matrix[sph, osph, imn, imunu] = wvfs.outgoing_separation_coefficient(munu[0], mn[0], munu[1],
                                                                                           mn[1], ps.k_fluid, distance)
    t_matrix2d = np.concatenate(np.concatenate(t_matrix, axis=1), axis=1)
    return t_matrix2d


def system_rhs(ps, order):
    rhs = np.zeros((ps.num_sph, (order+1)**2), dtype=complex)
    for sph in range(ps.num_sph):
        for mn in wvfs.multipoles(order):
            imn = mn[1]**2+mn[1]+mn[0]
            rhs[sph, imn] = wvfs.local_incident_coefficient(mn[0], mn[1], ps.k_fluid, ps.incident_field.dir,
                                                            ps.spheres[sph].pos, order)
    rhs1d = np.concatenate(rhs)
    return rhs1d


def solve_system(ps, order):
    sc_coef1d = scipy.linalg.solve(system_matrix(ps, order), system_rhs(ps, order))
    sc_coef = sc_coef1d.reshape((ps.num_sph, (order + 1) ** 2))
    in_coef = np.zeros((ps.num_sph, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        for mn in wvfs.multipoles(order):
            imn = mn[1]**2+mn[1]+mn[0]
            in_coef[sph, imn] = (scipy.special.spherical_jn(mn[1], ps.k_fluid * ps.spheres[sph].r) / scaled_coefficient(mn[1], sph, ps) +
                                 mths.sph_hankel1(mn[1], ps.k_fluid * ps.spheres[sph].r)) * sc_coef[sph, imn] / \
                                scipy.special.spherical_jn(mn[1], ps.k_spheres[sph] * ps.spheres[sph].r)

    # sol_coef = np.zeros((2 * ps.num_sph, (order+1)**2), dtype=complex)
    # for sph in range(ps.num_sph):
    #     sol_coef[2 * sph] = sc_coef[sph]
    #     sol_coef[2 * sph + 1] = in_coef[sph]
    # return sol_coef
    return sc_coef, in_coef


def effective_incident_coefficients(sph, sc_coef, ps, order):
    ef_inc_coef = np.zeros((order + 1) ** 2, dtype=complex)
    for mn in wvfs.multipoles(order):
        imn = mn[1]**2+mn[1]+mn[0]
        ef_inc_coef[imn] = sc_coef[imn] / scaled_coefficient(mn[1], sph, ps)
    return ef_inc_coef
