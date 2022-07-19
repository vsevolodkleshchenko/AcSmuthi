import numpy as np
import scipy.special
import classes as cls
import tsystem
import mathematics as mths
import wavefunctions as wvfs


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

    # print(s1, s2, np.linalg.det(s1), np.linalg.det(s2), sep='\n')

    return np.linalg.det(s1) / np.linalg.det(s2)


def scaled_coefficients(ps, order):
    scaled_coef = np.zeros((ps.num_sph, order + 1), dtype=complex)
    for sph in range(ps.num_sph):
        for n in range(order + 1):
            scaled_coef[sph, n] = scaled_coefficient(n, sph, ps)
    return scaled_coef


def local_scattered_coefficient(m, n, sph, ps, sol_coef, order):
    sc_coefficient_array = np.zeros(ps.num_sph * (order+1) ** 2, dtype=complex)
    i = 0
    for osph in np.where(np.arange(ps.num_sph) != sph)[0]:
        for munu in wvfs.multipoles(order):
            dist = ps.spheres[sph].pos - ps.spheres[osph].pos
            sc_coefficient_array[i] = sol_coef[2 * osph, munu[1] ** 2 + munu[1] + munu[0]] * \
                                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, ps.k_fluid,
                                                                                 dist)
            i += 1
    return mths.complex_fsum(sc_coefficient_array)


def effective_coefficient(m, n, sph, ps, sol_coef, order):
    return wvfs.local_incident_coefficient(m, n, ps.k_fluid, ps.incident_field.dir, ps.spheres[sph].pos, order) + \
              local_scattered_coefficient(m, n, sph, ps, sol_coef, order)


def effective_coefficients(ps, sol_coef, order):
    ef_coef = np.zeros((ps.num_sph, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        i = 0
        for mn in wvfs.multipoles(order):
            ef_coef[sph, i] = effective_coefficient(mn[0], mn[1], sph, ps, sol_coef, order)
            i += 1
    return ef_coef


def local_incident_coefficients(ps, order):
    l_inc_coef = np.zeros((ps.num_sph, (order + 1) ** 2), dtype=complex)
    for sph in range(ps.num_sph):
        i = 0
        for mn in wvfs.multipoles(order):
            l_inc_coef[sph, i] = wvfs.local_incident_coefficient(mn[0], mn[1], ps.k_fluid, ps.incident_field.dir,
                                                                 ps.spheres[sph].pos, order)
            i += 1
    return l_inc_coef


def test_function():
    # physical system
    ps = cls.build_ps_1s()

    # order of decomposition
    order = 10

    solution_coefficients = tsystem.solve_system(ps, order)
    scattered_coefficients = solution_coefficients[::2]
    eff_coefficients = effective_coefficients(ps, solution_coefficients, order)
    print(scattered_coefficients / eff_coefficients)

    scaled_coefficients_lopes = scaled_coefficients(ps, order)
    print(scaled_coefficients_lopes)


test_function()
