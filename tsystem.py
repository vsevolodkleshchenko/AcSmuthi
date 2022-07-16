import numpy as np
import scipy
import scipy.special
import wavefunctions as wvfs
import mathematics as mths
import classes as cls


def system_matrix(k, ro_fluid, positions, spheres, order):
    r""" Builds T^{-1} - matrix """
    k_fluid = mths.dec_to_sph(k[0], k[1], k[2])[0]
    num_of_coef = (order + 1) ** 2
    block_width = num_of_coef * 2
    block_height = num_of_coef * 2
    num_sph = len(spheres)
    t_matrix = np.zeros((block_height * num_sph, block_width * num_sph), dtype=complex)
    all_spheres = np.arange(num_sph)
    for sph in all_spheres:
        k_sph = spheres[sph, 0]
        r_sph = spheres[sph, 1]
        ro_sph = spheres[sph, 2]
        for n in range(order + 1):
            # diagonal block
            col_idx_1 = np.arange(sph * block_width + n ** 2, sph * block_width + (n + 1) ** 2)
            col_idx_2 = col_idx_1 + num_of_coef
            row_idx_1 = np.arange(sph * block_height + 2 * n ** 2, sph * block_height + 2 * (n + 1) ** 2, 2)
            row_idx_2 = np.arange(sph * block_height + 2 * n ** 2 + 1, sph * block_height + 2 * (n + 1) ** 2, 2)
            t_matrix[row_idx_1, col_idx_1] = - mths.sph_hankel1(n, k_fluid * r_sph)
            t_matrix[row_idx_2, col_idx_1] = - mths.sph_hankel1_der(n, k_fluid * r_sph)
            t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph * r_sph)
            t_matrix[row_idx_2, col_idx_2] = ro_fluid / ro_sph * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True)
            # not diagonal block
            other_sph = np.where(all_spheres != sph)[0]
            for osph in other_sph:
                for m in range(-n, n + 1):
                    for munu in wvfs.multipoles(order):
                        t_matrix[sph * block_height + 2 * (n ** 2 + n + m),
                                 osph * block_width + munu[1] ** 2 + munu[1] + munu[0]] = \
                            -scipy.special.spherical_jn(n, k_fluid * r_sph) * \
                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, k, positions[osph] - positions[sph])
                        t_matrix[sph * block_height + 2 * (n ** 2 + n + m) + 1,
                                 osph * block_width + munu[1] ** 2 + munu[1] + munu[0]] = \
                            -scipy.special.spherical_jn(n, k_fluid * r_sph, derivative=True) * \
                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, k, positions[osph] - positions[sph])
    return t_matrix


def system_rhs(k, pos, spheres, order):
    r""" build right hand side of system """
    k_fluid = mths.dec_to_sph(k[0], k[1], k[2])[0]
    num_sph = len(spheres)
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * num_sph, dtype=complex)
    for sph in range(num_sph):
        for mn in wvfs.multipoles(order):
            loc_inc_coef = wvfs.local_incident_coefficient(mn[0], mn[1], k, pos[sph], order)
            rhs[sph * 2 * num_of_coef + 2 * (mn[1] ** 2 + mn[1] + mn[0])] = loc_inc_coef * \
                                                                            scipy.special.spherical_jn(mn[1], k_fluid * spheres[sph, 1])
            rhs[sph * 2 * num_of_coef + 2 * (mn[1] ** 2 + mn[1] + mn[0]) + 1] = loc_inc_coef * \
                                                                                scipy.special.spherical_jn(mn[1], k_fluid * spheres[sph, 1], derivative=True)
    return rhs


def solve_system(k, ro_fluid, positions, spheres, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the spheres """
    num_sph = len(spheres)
    t_matrix = system_matrix(k, ro_fluid, positions, spheres, order)
    rhs = system_rhs(k, positions, spheres, order)
    solution_coefficients = scipy.linalg.solve(t_matrix, rhs)
    return np.array(np.split(solution_coefficients, 2 * num_sph))


########################################################################################################################


def system_matrix_cls(ps, order):
    r""" Builds T^{-1} - matrix """
    freq, k, k_fluid, ro_fluid, positions, spheres, p0, intensity, num_sph = cls.ps_to_param(ps)

    num_of_coef = (order + 1) ** 2
    block_width = num_of_coef * 2
    block_height = num_of_coef * 2
    num_sph = len(spheres)
    t_matrix = np.zeros((block_height * num_sph, block_width * num_sph), dtype=complex)
    all_spheres = np.arange(num_sph)
    for sph in all_spheres:
        k_sph = spheres[sph, 0]
        r_sph = spheres[sph, 1]
        ro_sph = spheres[sph, 2]
        for n in range(order + 1):
            # diagonal block
            col_idx_1 = np.arange(sph * block_width + n ** 2, sph * block_width + (n + 1) ** 2)
            col_idx_2 = col_idx_1 + num_of_coef
            row_idx_1 = np.arange(sph * block_height + 2 * n ** 2, sph * block_height + 2 * (n + 1) ** 2, 2)
            row_idx_2 = np.arange(sph * block_height + 2 * n ** 2 + 1, sph * block_height + 2 * (n + 1) ** 2, 2)
            t_matrix[row_idx_1, col_idx_1] = - mths.sph_hankel1(n, k_fluid * r_sph)
            t_matrix[row_idx_2, col_idx_1] = - mths.sph_hankel1_der(n, k_fluid * r_sph)
            t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph * r_sph)
            t_matrix[row_idx_2, col_idx_2] = ro_fluid / ro_sph * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True)
            # not diagonal block
            other_sph = np.where(all_spheres != sph)[0]
            for osph in other_sph:
                for m in range(-n, n + 1):
                    for munu in wvfs.multipoles(order):
                        t_matrix[sph * block_height + 2 * (n ** 2 + n + m),
                                 osph * block_width + munu[1] ** 2 + munu[1] + munu[0]] = \
                            -scipy.special.spherical_jn(n, k_fluid * r_sph) * \
                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, k, positions[osph] - positions[sph])
                        t_matrix[sph * block_height + 2 * (n ** 2 + n + m) + 1,
                                 osph * block_width + munu[1] ** 2 + munu[1] + munu[0]] = \
                            -scipy.special.spherical_jn(n, k_fluid * r_sph, derivative=True) * \
                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, k, positions[osph] - positions[sph])
    return t_matrix


def system_rhs_cls(ps, order):
    r""" build right hand side of system """
    freq, k, k_fluid, ro_fluid, positions, spheres, p0, intensity, num_sph = cls.ps_to_param(ps)

    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * num_sph, dtype=complex)
    for sph in range(num_sph):
        for mn in wvfs.multipoles(order):
            loc_inc_coef = wvfs.local_incident_coefficient(mn[0], mn[1], k, positions[sph], order)
            rhs[sph * 2 * num_of_coef + 2 * (mn[1] ** 2 + mn[1] + mn[0])] = loc_inc_coef * \
                                                                            scipy.special.spherical_jn(mn[1], k_fluid * spheres[sph, 1])
            rhs[sph * 2 * num_of_coef + 2 * (mn[1] ** 2 + mn[1] + mn[0]) + 1] = loc_inc_coef * \
                                                                                scipy.special.spherical_jn(mn[1], k_fluid * spheres[sph, 1], derivative=True)
    return rhs


def solve_system_cls(ps, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the spheres """
    freq, k, k_fluid, ro_fluid, positions, spheres, p0, intensity, num_sph = cls.ps_to_param(ps)

    t_matrix = system_matrix_cls(ps, order)
    rhs = system_rhs_cls(ps, order)
    solution_coefficients = scipy.linalg.solve(t_matrix, rhs)
    return np.array(np.split(solution_coefficients, 2 * num_sph))
