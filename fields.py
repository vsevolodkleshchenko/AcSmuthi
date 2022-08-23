import reflection
import wavefunctions as wvfs
import mathematics as mths
import numpy as np


def reflected_field(x, y, z, reflected_coefficients, ps, order):
    r"""Counts reflected scattered field on mesh x, y, z"""
    ref_field_array = np.split(np.repeat(reflected_coefficients, len(x)), (order + 1) ** 2) * \
                      wvfs.regular_wvfs_array(order, x, y, z, ps.k_fluid)
    ref_field = mths.multipoles_fsum(ref_field_array, len(x))
    return ref_field


def scattered_field(x, y, z, scattered_coefficients, ps, order):
    r"""Counts scattered field on mesh x, y, z"""
    sc_field_array = np.zeros((ps.num_sph, (order + 1) ** 2, len(x)), dtype=complex)
    for sph in range(ps.num_sph):
        sph_sc_coef = np.split(np.repeat(scattered_coefficients[sph], len(x)), (order + 1) ** 2)
        sc_field_array[sph] = wvfs.outgoing_wvfs_array(order, x - ps.spheres[sph].pos[0], y - ps.spheres[sph].pos[1],
                                                       z - ps.spheres[sph].pos[2], ps.k_fluid) * sph_sc_coef
    # return np.sum(tot_field_array, axis=(0, 1))
    return mths.spheres_multipoles_fsum(sc_field_array, len(x))


def incident_field(x, y, z, ps, order):
    r"""Counts effective incident field on mesh x, y, z"""
    # inc_field_array = wvfs.incident_coefficients_array(ps.incident_field.dir, len(x), order) * \
    #                   wvfs.regular_wvfs_array(order, x, y, z, ps.k_fluid)
    # inc_field = mths.multipoles_fsum(inc_field_array, len(x))
    direct = ps.incident_field.dir
    inc_field = np.exp(1j * ps.k_fluid * (x * direct[0] + y * direct[1] + z * direct[2]))

    if ps.interface:
        ref_direct = reflection.reflection_dir(direct, ps.interface.normal)
        h = ps.k_fluid * np.sqrt(1 - np.dot(ps.incident_field.dir, ps.interface.normal) ** 2)
        ref_coef = reflection.ref_coef_h(h, ps.omega, ps.fluid.speed, ps.interface.speed, ps.fluid.rho,
                                         ps.interface.rho)
        image_o = - 2 * ps.interface.normal * ps.interface.int_dist0
        inc_field += ref_coef * np.exp(1j * ps.k_fluid * ((x - image_o[0]) * ref_direct[0] + (y - image_o[1]) *
                                                          ref_direct[1] + (z - image_o[2]) * ref_direct[2]))
    return inc_field


def inner_field(x, y, z, inner_coefficients, tot_field, ps, order):
    r"""Counts fields inside spheres and add it to total field on mesh x, y, z"""
    in_field_array = np.zeros((ps.num_sph, (order + 1) ** 2, len(x)), dtype=complex)
    in_field = np.zeros((ps.num_sph, len(x)), dtype=complex)
    for sph in range(ps.num_sph):
        sph_in_coef = np.split(np.repeat(inner_coefficients[sph], len(x)), (order + 1) ** 2)
        in_field_array[sph] = wvfs.regular_wvfs_array(order, x - ps.spheres[sph].pos[0], y - ps.spheres[sph].pos[1],
                                                        z - ps.spheres[sph].pos[2], ps.k_spheres[sph]) * sph_in_coef
        in_field[sph] = mths.multipoles_fsum(in_field_array[sph], len(x))
    for sph in range(ps.num_sph):
        rx, ry, rz = x - ps.spheres[sph].pos[0], y - ps.spheres[sph].pos[1], z - ps.spheres[sph].pos[2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        tot_field = np.where(r <= ps.spheres[sph].r, in_field[sph], tot_field)
    return tot_field


def total_field(x, y, z, solution_coefficients, ps, order, incident_field_on=False):
    r"""Counts total field on mesh x, y, z"""
    tot_field = np.zeros(len(x), dtype=complex)

    if ps.interface:
        incident_coefs, scattered_coefs, inner_coefs, reflected_coefs, local_reflected_coefs = solution_coefficients
        tot_field += reflected_field(x, y, z, reflected_coefs, ps, order)
    else:
        incident_coefs, scattered_coefs, inner_coefs = solution_coefficients

    tot_field += scattered_field(x, y, z, scattered_coefs, ps, order)

    if incident_field_on:
        tot_field += incident_field(x, y, z, ps, order)

    tot_field = inner_field(x, y, z, inner_coefs, tot_field, ps, order)

    return tot_field
