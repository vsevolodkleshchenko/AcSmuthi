import numpy as np
import scipy
import scipy.special as ss
import wavefunctions as wvfs
import mathematics as mths
import reflection
import time


def scaled_coefficient(n, sph, ps):
    r"""Scaled coefficient for spheres[sph]"""
    k_q = ps.k_spheres[sph]
    rho_0 = ps.fluid.rho
    k = ps.k_fluid
    rho_q = ps.spheres[sph].rho
    a_q = ps.spheres[sph].r
    gamma_q = k_q * rho_0 / k / rho_q
    s1 = np.zeros((2, 2), dtype=complex)
    s2 = np.zeros((2, 2), dtype=complex)

    s1[0, 0] = gamma_q * ss.spherical_jn(n, k * a_q)
    s1[0, 1] = ss.spherical_jn(n, k_q * a_q)
    s1[1, 0] = ss.spherical_jn(n, k * a_q, derivative=True)
    s1[1, 1] = ss.spherical_jn(n, k_q * a_q, derivative=True)

    s2[0, 0] = - gamma_q * mths.sph_hankel1(n, k * a_q)
    s2[0, 1] = ss.spherical_jn(n, k_q * a_q)
    s2[1, 0] = - mths.sph_hankel1_der(n, k * a_q)
    s2[1, 1] = ss.spherical_jn(n, k_q * a_q, derivative=True)

    return np.linalg.det(s1) / np.linalg.det(s2)


def system_matrix(ps, order):
    r"""Build system matrix (T^(-1))"""
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
    return t_matrix2d


def system_rhs(ps, order):
    r"""Build right hand side of system"""
    return np.dot(d_matrix(ps, order), wvfs.incident_coefficients(ps.incident_field.dir, order))


def solve_system(ps, order):
    r"""Solve linear system for scattering on spheres"""
    inc_coef = system_rhs(ps, order)
    t_ = system_matrix(ps, order)
    sc_coef1d = scipy.linalg.solve(t_, inc_coef)
    sc_coef = sc_coef1d.reshape((ps.num_sph, (order + 1) ** 2))
    in_coef = inner_coefficients(sc_coef, ps, order)
    return inc_coef, sc_coef, in_coef


def effective_incident_coefficients(sph, sc_coef, ps, order):
    r"""Build np.array of effective incident coefficients for all n <= order """
    ef_inc_coef = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        ef_inc_coef[imn] = sc_coef[imn] / scaled_coefficient(n, sph, ps)
    return ef_inc_coef


def d_matrix(ps, order):
    r"""Build D matrix - matrix of S coefficients that translate origin incident coefficients to local"""
    d = np.zeros((ps.num_sph, (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(order):
                imunu = nu ** 2 + nu + mu
                d[sph, imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, ps.k_fluid, ps.spheres[sph].pos)
    d_2d = d.reshape((ps.num_sph * (order + 1) ** 2, (order + 1) ** 2))
    return d_2d


def r_matrix(ps, order, order_approx=1):
    r"""Build R matrix - reflection matrix"""
    r = np.zeros((ps.num_sph, (order + 1) ** 2, (order + 1) ** 2), dtype=complex)
    a, alpha = reflection.ref_coef_approx(ps.omega, ps.fluid.speed, ps.interface.speed, ps.fluid.rho,
                                          ps.interface.rho, order_approx, -3)
    for m, n in wvfs.multipoles(order):
        imn = n ** 2 + n + m
        for mu, nu, in wvfs.multipoles(order):
            imunu = nu ** 2 + nu + mu
            for sph in range(ps.num_sph):
                image_poses = reflection.image_poses(ps.spheres[sph], ps.interface, alpha)
                image_contribution = reflection.image_contribution(m, n, mu, nu, ps.k_fluid, image_poses, a)
                r[sph, imn, imunu] = (-1) ** (nu + mu) * image_contribution
    r_2d = np.concatenate(r, axis=1)
    return r_2d


def inner_coefficients(sc_coef, ps, order):
    r"""Counts coefficients of decompositions fields inside spheres"""
    in_coef = np.zeros((ps.num_sph, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            in_coef[sph, imn] = (ss.spherical_jn(n, ps.k_fluid * ps.spheres[sph].r) / scaled_coefficient(n, sph, ps) +
                                 mths.sph_hankel1(n, ps.k_fluid * ps.spheres[sph].r)) * sc_coef[sph, imn] / \
                                ss.spherical_jn(n, ps.k_spheres[sph] * ps.spheres[sph].r)
    return in_coef


def layer_inc_coef_origin(ps, order):
    r"""Effective incident coefficients - coefficients of decomposition incident field and it's reflection"""
    inc_coef_origin = np.zeros((order + 1) ** 2, dtype=complex)
    for m, n in wvfs.multipoles(order):
        ref_dir = reflection.reflection_dir(ps.incident_field.dir, ps.interface.normal)
        image_o = - 2 * ps.interface.normal * ps.interface.int_dist0
        inc_coef_origin[n ** 2 + n + m] = wvfs.incident_coefficient(m, n, ps.incident_field.dir) + \
                                          wvfs.local_incident_coefficient(m, n, ps.k_fluid, ref_dir, -image_o, order)
    return inc_coef_origin


def solve_layer_system(ps, order):
    r"""Find solution coefficients of scattering on spheres with interface(layer)"""
    t = np.linalg.inv(system_matrix(ps, order))
    d = d_matrix(ps, order)
    r = r_matrix(ps, order)

    inc_coef_origin = layer_inc_coef_origin(ps, order)
    local_inc_coefs = np.dot(d, inc_coef_origin)

    m1 = t @ d
    m2 = r @ m1
    m3 = np.linalg.inv(np.eye(m2.shape[0]) - m2)

    sc_coef1d = np.dot(m1 @ m3, inc_coef_origin)
    sc_coef = sc_coef1d.reshape((ps.num_sph, (order + 1) ** 2))

    ref_coef = np.dot(m3, inc_coef_origin) - inc_coef_origin
    local_ref_coefs = np.dot(d, ref_coef)

    in_coef = inner_coefficients(sc_coef, ps, order)
    return local_inc_coefs, sc_coef, in_coef, ref_coef, local_ref_coefs
