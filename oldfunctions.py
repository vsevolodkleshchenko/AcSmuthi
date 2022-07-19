from sympy.physics.wigner import wigner_3j


def yz_old(span, plane_number, k, ro, pos, spheres, order):
    r"""
    OLD 2D heat-plot in YZ plane for x[plane_number]
    --->z """
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    tot_field = np.real(total_field(x, y, z, k, ro, pos, spheres, order))

    for sph in range(len(spheres)):
        x_min = pos[sph, 0] - spheres[sph, 1]
        y_min = pos[sph, 1] - spheres[sph, 1]
        z_min = pos[sph, 2] - spheres[sph, 1]
        x_max = pos[sph, 0] + spheres[sph, 1]
        y_max = pos[sph, 1] + spheres[sph, 1]
        z_max = pos[sph, 2] + spheres[sph, 1]
        tot_field = np.where((x >= x_min) & (x <= x_max) &
                             (y >= y_min) & (y <= y_max) &
                             (z >= z_min) & (z <= z_max), 0, tot_field)

    yz = np.asarray(tot_field[(plane_number - 1) * len(span_y) * len(span_z):
                              (plane_number - 1) * len(span_y) * len(span_z) +
                              len(span_y) * len(span_z)]).reshape(len(span_y), len(span_z))
    fig, ax = plt.subplots()
    plt.xlabel('z axis')
    plt.ylabel('y axis')
    im = ax.imshow(yz, cmap='seismic', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_y.min(), span_y.max()])
    plt.colorbar(im)
    plt.show()


def syst_rhs_old(k, spheres, order):
    r""" build right hand side of system from spheres.pdf"""
    k_abs = dec_to_sph(k[0], k[1], k[2])[0]
    num_of_sph = len(spheres)
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * num_of_sph, dtype=complex)
    for n in range(order + 1):
        idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        m = np.arange(-n, n + 1)
        for sph in range(num_of_sph):
            rhs[idx_1] = inc_coef(m, n, k) * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1])
            rhs[idx_2] = inc_coef(m, n, k) * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1], derivative=True)
            idx_1 += num_of_coef * 2
            idx_2 += num_of_coef * 2
    return rhs


def old_total_field(x, y, z, k, ro, pos, spheres, order):
    r""" counts field outside the spheres """
    coef = syst_solve(k, ro, pos, spheres, order)
    tot_field_array = np.zeros((len(spheres), (order + 1) ** 2, len(x)), dtype=complex)
    tot_field = np.zeros(len(x), dtype=complex)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            for sph in range(len(spheres)):
                tot_field += coef[2 * sph, n ** 2 + n + m] * \
                             outgoing_wvfs(m, n, x - pos[sph][0], y - pos[sph][1], z - pos[sph][2], k)
            # tot_field += inc_coef(m, n, k) * regular_wvfs(m, n, x, y, z, k)
                # tot_field += local_inc_coef(m, n, k) * regular_wvfs(m, n, x, y, z, k)
    return tot_field # np.abs(tot_field - tot_field1)


def old_gaunt_coef(n, m, nu, mu, q):
    r""" Gaunt coefficient: G(n,m;nu,mu;q)
    eq(3.71) in 'Encyclopedia' """
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    return (-1) ** (m + mu) * s * float(wigner_3j(n, nu, q, 0, 0, 0)) * \
           float(wigner_3j(n, nu, q, m, mu, - m - mu))


def old_local_inc_coef(m, n, k, sph_pos, order):
    r""" Counts local incident coefficients
    d^m_nj - eq(42) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    inccoef = 0
    for nu in range(order + 1):
        for mu in range(-nu, nu + 1):
            inccoef += inc_coef(mu, nu, k) * sepc_matr_coef(mu, m, nu, n, k, sph_pos)
    return inccoef


def old_sepc_matr_coef(m, mu, n, nu, k, dist):
    r""" Coefficient ^S^mmu_nnu(b) of separation matrix
    eq(3.92) and eq(3.74) in 'Encyclopedia' """
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    sum = 0
    for q in range(0, q_lim + 1, 2):
        sum += (-1) ** q * regular_wvfs(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
               gaunt_coef(n, m, nu, -mu, q0 + 2 * q)
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * sum


def old_build_slice_xz(span, plane_number):
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
    y, x, z = grid[:, 0], grid[:, 1], grid[:, 2]

    x_p = x[(plane_number - 1) * len(span_x) * len(span_z):
            (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
    y_p = y[(plane_number - 1) * len(span_x) * len(span_z):
            (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
    z_p = z[(plane_number - 1) * len(span_x) * len(span_z):
            (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
    return x_p, y_p, z_p


def old_xz_plot(span, plane_number, k, ro, pos, spheres, order):
    r""" Count field and build a 2D heat-plot in XZ plane for span_y[plane_number]
    --->z """
    span_x, span_y, span_z = span[0], span[1], span[2]
    x_p, y_p, z_p = build_slice_xz(span, plane_number)
    tot_field = np.real(total_field(x_p, y_p, z_p, k, ro, pos, spheres, order))
    tot_field = draw_spheres(tot_field, pos, spheres, x_p, y_p, z_p)
    xz = tot_field.reshape(len(span_x), len(span_z))
    fig, ax = plt.subplots()
    plt.xlabel('z axis')
    plt.ylabel('x axis')
    im = ax.imshow(xz, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_x.min(), span_x.max()])
    plt.colorbar(im)
    plt.show()


def old_cross_section(k, ro, pos, spheres, order):
    r""" Counts scattering and extinction cross sections Sigma_sc and Sigma_ex
    eq(46,47) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    coef = syst_solve(k, ro, pos, spheres, order)
    num_sph = len(pos)
    sigma_ex, sigma_sc1, sigma_sc2 = 0, 0, 0
    for j in range(num_sph):
        for n in range(order + 1):
            for m in range(-n, n + 1):
                for l in range(num_sph):
                    for nu in range(order + 1):
                        for mu in range(-nu, nu + 1):
                            sigma_sc2 += np.conj(coef[2 * j, n ** 2 + n + m]) * \
                                       coef[2 * l, nu ** 2 + nu + mu] * \
                                       sepc_matr_coef(mu, m, nu, n, k, pos[j] - pos[l])
                sigma_sc1 += np.abs(coef[2 * j, n ** 2 + n + m])
                sigma_ex += - np.real(coef[2 * j, n ** 2 + n + m] * np.conj(inc_coef(m, n, k)))
    sigma_sc = np.real(sigma_sc1 + sigma_sc2)
    return sigma_sc, sigma_ex


def system_matrix_old(ps, order):
    r""" Builds T^{-1} - matrix """
    num_of_coef = (order + 1) ** 2
    block_width = num_of_coef * 2
    block_height = num_of_coef * 2
    t_matrix = np.zeros((block_height * ps.num_sph, block_width * ps.num_sph), dtype=complex)
    all_spheres = np.arange(ps.num_sph)
    for sph in all_spheres:
        k_sph = ps.k_spheres[sph]
        r_sph = ps.spheres[sph].r
        ro_sph = ps.spheres[sph].rho
        for n in range(order + 1):
            # diagonal block
            col_idx_1 = np.arange(sph * block_width + n ** 2, sph * block_width + (n + 1) ** 2)
            col_idx_2 = col_idx_1 + num_of_coef
            row_idx_1 = np.arange(sph * block_height + 2 * n ** 2, sph * block_height + 2 * (n + 1) ** 2, 2)
            row_idx_2 = np.arange(sph * block_height + 2 * n ** 2 + 1, sph * block_height + 2 * (n + 1) ** 2, 2)
            t_matrix[row_idx_1, col_idx_1] = - mths.sph_hankel1(n, ps.k_fluid * r_sph)
            t_matrix[row_idx_2, col_idx_1] = - mths.sph_hankel1_der(n, ps.k_fluid * r_sph) * ps.k_fluid
            t_matrix[row_idx_1, col_idx_2] = scipy.special.spherical_jn(n, k_sph * r_sph)
            t_matrix[row_idx_2, col_idx_2] = ps.fluid.rho / ro_sph * k_sph * scipy.special.spherical_jn(n, k_sph * r_sph, derivative=True)
            # not diagonal block
            other_sph = np.where(all_spheres != sph)[0]
            for osph in other_sph:
                for m in range(-n, n + 1):
                    for munu in wvfs.multipoles(order):
                        t_matrix[sph * block_height + 2 * (n ** 2 + n + m),
                                 osph * block_width + munu[1] ** 2 + munu[1] + munu[0]] = \
                            -scipy.special.spherical_jn(n, ps.k_fluid * r_sph) * \
                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, ps.k_fluid,
                                                                 ps.spheres[osph].pos - ps.spheres[sph].pos)
                        t_matrix[sph * block_height + 2 * (n ** 2 + n + m) + 1,
                                 osph * block_width + munu[1] ** 2 + munu[1] + munu[0]] = \
                            -scipy.special.spherical_jn(n, ps.k_fluid * r_sph, derivative=True) * ps.k_fluid * \
                            wvfs.outgoing_separation_coefficient(munu[0], m, munu[1], n, ps.k_fluid,
                                                                 ps.spheres[osph].pos - ps.spheres[sph].pos)
    return t_matrix


def system_rhs_old(ps, order):
    r""" build right hand side of system """
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * ps.num_sph, dtype=complex)
    for sph in range(ps.num_sph):
        for mn in wvfs.multipoles(order):
            loc_inc_coef = wvfs.local_incident_coefficient(mn[0], mn[1], ps.k_fluid, ps.incident_field.dir,
                                                           ps.spheres[sph].pos, order)
            rhs[sph*2*num_of_coef + 2*(mn[1]**2+mn[1]+mn[0])] = loc_inc_coef * \
                                                                scipy.special.spherical_jn(mn[1], ps.k_fluid * ps.spheres[sph].r)
            rhs[sph*2*num_of_coef + 2*(mn[1]**2+mn[1]+mn[0])+1] = loc_inc_coef * ps.k_fluid * \
                                                                  scipy.special.spherical_jn(mn[1], ps.k_fluid * ps.spheres[sph].r, derivative=True)
    return rhs


def solve_system_old(ps, order):
    r""" solve T matrix system and counts a coefficients in decomposition
    of scattered field and field inside the spheres """
    t_matrix = system_matrix_old(ps, order)
    rhs = system_rhs_old(ps, order)
    solution_coefficients = scipy.linalg.solve(t_matrix, rhs)
    return np.array(np.split(solution_coefficients, 2 * ps.num_sph))


def total_field_m_old(x, y, z, k, ro, pos, spheres, order, m=-1):
    r""" Counts field outside the spheres for mth harmonic """
    coef = tsystem.solve_system(k, ro, pos, spheres, order)
    tot_field = 0
    for n in range(abs(m), order + 1):
        for sph in range(len(spheres)):
            tot_field += coef[2 * sph][n ** 2 + n + m] * \
                         wvfs.outgoing_wave_function(m, n, x - pos[sph][0], y - pos[sph][1], z - pos[sph][2], k)
        tot_field += wvfs.incident_coefficient(m, n, k) * wvfs.regular_wave_function(m, n, x, y, z, k)
    return tot_field


# a = np.arange(36)
# b = a.reshape((3, 3, 2, 2))
# c = a.reshape((4, 9))
# d = np.concatenate(np.concatenate(b, axis=1), axis=1)
# e = np.concatenate(c)
# print(a, b[0, 2, 0, 1], b, c, d, e, sep="\n")